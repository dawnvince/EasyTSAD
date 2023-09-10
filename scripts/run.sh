#!/bin/bash

# How to RUN the script:
# sh ./scripts/run.sh YOUR_CUDA_DEVICE METHOD_NAME

CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --task_mode one_by_one
CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --task_mode all_in_one
CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --task_mode transfer_within_dataset