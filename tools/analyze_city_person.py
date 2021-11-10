
import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.cityPerson.eval_demo import validate
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np


def single_gpu_test(model, data_loader, show=False, save_img=False, save_img_dir=''):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg, save_result=save_img,
                                     result_name=save_img_dir + '/' + str(i) + '.jpg')

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('checkpoint_start', type=int, default=1)
    parser.add_argument('checkpoint_end', type=int, default=100)
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save_img', action='store_true', help='save result image')
    parser.add_argument('--save_img_dir', type=str, help='the dir for result image', default='')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    work_dir = "/".join(args.checkpoint.split("/")[:-1] + ['runs'])
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    cfg = mmcv.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    new_test = True
    traditional_test = True
    val_train = False

    if new_test:
        nval_writer = SummaryWriter(work_dir + '/mr_val')
        ntrain_writer = SummaryWriter(work_dir + '/mr_train')

    if traditional_test:
        tval_writer = SummaryWriter(work_dir + '/lamr_val')
        ttrain_writer = SummaryWriter(work_dir + '/lamr_train')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
    trainval_conf = cfg.data.test.copy()
    trainval_conf.ann_file = 'datasets/CityPersons/train_vis.json'
    trainval_conf.img_prefix = cfg.data.train.img_prefix
    dataset = build_dataset(cfg.data.test)
    trainval_dataset = build_dataset(trainval_conf)
    #data_loader = build_dataloader(
    #    dataset,
    #    imgs_per_gpu=1,
    #    workers_per_gpu=0,
    #    dist=distributed,
    #    shuffle=False, drop_last=True)
    #trainval_dataloader = build_dataloader(
    #    trainval_dataset,
    #    imgs_per_gpu=1,
    #    workers_per_gpu=0,
    #    dist=distributed,
    #    shuffle=False, drop_last=True)


    if args.out is not None and not args.out.endswith(('.json', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    for i in range(args.checkpoint_start, args.checkpoint_end):
        #val_writer = SummaryWriter('runs/val')
        #train_writer = SummaryWriter('runs/train')
        #cfg = mmcv.Config.fromfile(args.config)
        # set cudnn_benchmark
        #if cfg.get('cudnn_benchmark', False):
        #    torch.backends.cudnn.benchmark = True
        #cfg.model.pretrained = None
        #cfg.data.test.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        #if args.launcher == 'none':
        #    distributed = False
        #else:
        #    distributed = True
        #    init_dist(args.launcher, **cfg.dist_params)
        #distributed = True
        #init_dist(args.launcher, **cfg.dist_params)
        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        #rank, world_size = get_dist_info()
        #trainval_conf = cfg.data.test.copy()
        #trainval_conf.ann_file = 'datasets/CityPersons/train_vis.json'
        #trainval_conf.img_prefix = cfg.data.train.img_prefix
        #dataset = build_dataset(cfg.data.test)
        #trainval_dataset = build_dataset(trainval_conf)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=distributed,
            shuffle=False)
        if val_train:
            trainval_dataloader = build_dataloader(
                trainval_dataset,
                imgs_per_gpu=12,
                workers_per_gpu=1,
                dist=distributed,
                shuffle=False, drop_last=True)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if not args.mean_teacher:
            while not osp.exists(args.checkpoint + str(i) + '.pth'):
                time.sleep(5)
            while i + 1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i + 1) + '.pth'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth', map_location='cpu')
            model.CLASSES = dataset.CLASSES
        else:
            while not osp.exists(args.checkpoint + str(i) + '.pth.stu'):
                time.sleep(5)
            while i + 1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i + 1) + '.pth.stu'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth.stu', map_location='cpu')
            checkpoint['meta'] = dict()
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES  # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.save_img, args.save_img_dir)
            if val_train:
                print("")
                outputs_trainval = single_gpu_test(model, trainval_dataloader, args.show, args.save_img, args.save_img_dir)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)
            if val_train:
                print("")
                outputs_trainval = multi_gpu_test(model, trainval_dataloader, args.tmpdir)

        print("processing...")
        def load_ann(ann_file):

            

            HtRng = [[50, 1e5 ** 2], [50, 75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
            VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2, 0.65], [0.2, 1e5 ** 2]]
            settings = [dict(h=HtRng[i], v=VisRng[i]) for i in range(4)]

            with open(ann_file) as t:
                ann = json.load(t)
            
            annotations = {}
            ann_count = 0
            keys = ['r', 's', 'h', 'a']
            set_id_ann_map = dict(r={}, s={}, h={}, a={})
            for anb in ann['annotations']:
                #print(anb["image_id"])
                ann_count += 1
                if anb['category_id'] == 1 and anb['height'] >= 20:
                    for j in range(4):
                        if settings[j]['h'][0] <= anb['height'] <= settings[j]['h'][1] and settings[j]['v'][0] <= anb['vis_ratio'] <= settings[j]['v'][1]:
                            if not anb['image_id'] in set_id_ann_map[keys[j]]:
                                set_id_ann_map[keys[j]][anb['image_id']] = []
                            if not anb['image_id'] in annotations:
                                annotations[anb['image_id']] = {}
                            set_id_ann_map[keys[j]][anb['image_id']].append(anb['id'])
                            annotations[anb['image_id']][anb["id"]] = anb["bbox"]

            return set_id_ann_map, annotations

        def crunch_mrs(imap, hits_map):
            mrs = {}
            for k, v in imap.items():
                hits = 0
                miss = 0
                for im_id, ann_list in v.items():
                    for ann_id in ann_list:
                        if hits_map[ann_id]:
                            hits += 1
                        else:
                            miss += 1
                mrs[k] = miss/(hits+miss)
            return mrs

        def find_hits(anns, _res):
            hit_map = {}
            for img_id, image_anns in anns.items():
                if img_id not in _res:
                    for ann_id, ann in image_anns.items():
                        hit_map[ann_id] = False
                    continue
                img_res = _res[img_id]
                #print("Looking hits for ", img_id)
                for ann_id, ann in image_anns.items():

                    max_ind = -1
                    max_iou = 0.0

                    gx1, gy1, gw, gh = ann
                    gt_area = gw * gh
                    gx2, gy2 = gx1 + gw, gy1 + gh
                    taken = []
                    for ind in range(len(img_res)):
                        if ind in taken:
                            continue
                        #print(ind, img_res[ind])
                        px1, py1, pw, ph, s = img_res[ind]
                        pred_area = pw * ph
                        px2, py2 = px1 + pw, py1 + ph

                        ix1 = max(gx1, px1)
                        ix2 = min(gx2, px2)
                        iy1 = max(gy1, py1)
                        iy2 = min(gy2, py2)
                        ih = iy2 - iy1
                        iw = ix2 - ix1
                        if ih > 0 and iw > 0:
                            intersection_area = ih * iw
                            
                            ji = intersection_area/(gt_area + pred_area - intersection_area)
                            if ji >= 0.5 or max_ind == -1:
                                max_ind = ind
                                max_iou = ji

                    if max_iou >= 0.5:
                        taken.append(max_ind)
                        hit_map[ann_id] = True
                    else:
                        hit_map[ann_id] = False
            return hit_map
        if rank == 0:
            res = {}
            check = []
            pred_count = 0
            for out in outputs:
                for s_out in out:
                    boxes, id = s_out
                    
                    if type(boxes) == list:
                        boxes = boxes[0]
                    boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                    if boxes is None:
                        boxes = []
                    if new_test:
                        pred_count += len(boxes)
                        res[id] = boxes
                    if len(boxes) > 0 and traditional_test:
                        for box in boxes:
                            temp = dict()
                            temp['image_id'] = id
                            temp['category_id'] = 1
                            temp['bbox'] = box[:4].tolist()
                            temp['score'] = float(box[4])
                            check.append(temp)
            if traditional_test:
                with open(args.out, 'w') as f:
                    json.dump(check, f)
                MRs = validate('datasets/CityPersons/val_gt.json', args.out)
                print("Checkpoint %d: [VR: %.2f], [VS: %.2f], [VH: %.2f], [VA: %.2f]" % (i, MRs[0], MRs[1], MRs[2], MRs[3]))
                tval_writer.add_scalar('Reasonable', MRs[0], i)
                tval_writer.add_scalar('Small', MRs[1], i)
                tval_writer.add_scalar('Heavy', MRs[2], i)
                tval_writer.add_scalar('All', MRs[3], i)
                
                tval_writer.flush()

            if new_test:
                #print("Gathered results...")
                info_map, test_anns = load_ann(cfg.data.test.ann_file)
                #print("Loaded anns\nfinding hits")
                hit_map = find_hits(test_anns, res)
                tps = np.sum([1 for k, v in hit_map.items() if v])
                fps = pred_count - tps
                fppi = fps/len(dataset)
                print("Crunching MRs")
                mrs = crunch_mrs(info_map, hit_map)
                nval_writer.add_scalar('Reasonable', mrs['r'], i)
                nval_writer.add_scalar('Small', mrs['s'], i)
                nval_writer.add_scalar('Heavy', mrs['h'], i)
                nval_writer.add_scalar('All', mrs['a'], i)
                nval_writer.add_scalar('FPPI', fppi, i)
                nval_writer.flush()
                print("Checkpoint %d: [VR: %.2f], [VS: %.2f], [VH: %.2f], [VA: %.2f], [FPPI: %.2f]" % (i, mrs['r'], mrs['s'], mrs['h'], mrs['a'], fppi))
            if not val_train:
                continue
            res = {}
            check = []
            pred_count = 0
            for out in outputs_trainval:
                for s_out in out:
                    boxes, id = s_out

                    if type(boxes) == list:
                        boxes = boxes[0]
                    boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                    if boxes is None:
                        boxes = []
                    if new_test:
                        pred_count += len(boxes)
                        res[id] = boxes
                    if len(boxes) > 0 and traditional_test:
                        for box in boxes:
                            temp = dict()
                            temp['image_id'] = id
                            temp['category_id'] = 1
                            temp['bbox'] = box[:4].tolist()
                            temp['score'] = float(box[4])
                            check.append(temp)
            if traditional_test:
                with open(args.out, 'w') as f:
                    json.dump(check, f)
                MRs = validate('datasets/CityPersons/train_vis.json', args.out)
                print("Checkpoint %d: [tR: %.2f], [tS: %.2f], [tH: %.2f], [tA: %.2f]" % (i, MRs[0], MRs[1], MRs[2], MRs[3]))
                ttrain_writer.add_scalar('Reasonable', MRs[0], i)
                ttrain_writer.add_scalar('Small', MRs[1], i)
                ttrain_writer.add_scalar('Heavy', MRs[2], i)
                ttrain_writer.add_scalar('All', MRs[3], i)
                                           
                ttrain_writer.flush()

            if new_test:
                #print("Gathered results...")
                info_map, trainval_anns = load_ann(trainval_conf.ann_file)
                #print("Loaded anns\nfinding hits")
                hit_map = find_hits(trainval_anns, res)
                tps = np.sum([1 for k, v in hit_map.items() if v])
                fps = pred_count - tps
                fppi = fps/len(trainval_dataset)
                print("Crunching MRs")
                mrs = crunch_mrs(info_map, hit_map)
                ntrain_writer.add_scalar('Reasonable', mrs['r'], i)
                ntrain_writer.add_scalar('Small', mrs['s'], i)
                ntrain_writer.add_scalar('Heavy', mrs['h'], i)
                ntrain_writer.add_scalar('All', mrs['a'], i)
                ntrain_writer.flush()
                print("Checkpoint %d: [tR: %.2f], [tS: %.2f], [tH: %.2f], [tA: %.2f], [FPPI: %.2f]" % (i, mrs['r'], mrs['s'], mrs['h'], mrs['a'], fppi))


if __name__ == '__main__':
    main()
