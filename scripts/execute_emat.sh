#!/usr/bin/env bash
set -euo pipefail

# Config dataset
DATASET="BGL"
DATA_PATH="dataset/BGL"

# example of executing single EM-AT model
python main.py --anormly_ratio 1 --win_size 100 --num_epochs 3 --e_layer_num 3  --data_seq_len 10 --k 5 --batch_size 64 --mode train \
    --dataset "${DATASET}" --data_path "${DATA_PATH}" --input_c 10 --output_c 10

python main.py --anormly_ratio 1 --win_size 100 --num_epochs 3 --e_layer_num 3  --data_seq_len 10 --k 5 --batch_size 64 --mode test \
    --dataset "${DATASET}" --data_path "${DATA_PATH}" --input_c 10 --output_c 10