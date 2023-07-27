import argparse
import os
import os.path as osp
import sys
import cv2
import torch
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from mmdet.apis import inference_detector, init_detector, show_result 

#funçao que lida com os argumentos passados pela linha de comando
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_video', type=str, help='the path of input video')
    parser.add_argument('output_dir', type=str, help='the dir for result frames')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args

#funçao que realiza a detecçao de objetos em cada frame do vídeo
def mock_detector(model, video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    frame_count = 0 #inicializa contador de frames
    while True:
        ret, frame = cap.read() #le um frame do video
        if not ret:
            break
        results = inference_detector(model, frame)
        result_name = f'frame_{frame_count}_result.jpg' #define o nome do arquivo de resultado
        result_name = os.path.join(output_dir, result_name) #define o caminho completo para o arquivo de resultado
        show_result(frame, results, model.CLASSES, out_file=result_name) #exibe o resultado e salva o frame
        frame_count += 1 #incrementa o contador de frames

#funçao principal que inicializa o detector e chama mock_detector
def run_detector_on_dataset():
    args = parse_args() #parseia os argumentos da linha de comando
    output_dir = args.output_dir #obtém o diretório de saída dos argumentos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) #cria o diretorio de saída se ele nao existir

    #inicializa o modelo detector
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    mock_detector(model, args.input_video, output_dir) #chama a funçao que realiza a detecçao de objetos em cada frame do vídeo

#se este script for executado diretamente (em vez de importado), chama a funçao principal
if __name__ == '__main__':
    run_detector_on_dataset()
