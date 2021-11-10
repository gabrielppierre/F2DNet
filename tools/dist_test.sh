#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
EP_PATH=$2
EP_SET=$3
EP_END=$4
GPUS=$5

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/test_city_person.py $CONFIG $EP_PATH $EP_SET $EP_END --launcher pytorch ${@:6}
