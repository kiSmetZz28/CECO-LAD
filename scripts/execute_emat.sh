#!/usr/bin/env bash
set -euo pipefail

# Config dataset
DATASET="BGL"
DATA_PATH="dataset/BGL"

# example of executing EM-AT model   # num_epochs, k, e_layer_num, batch_size
python main.py --anormly_ratio 1 --win_size 100 --num_epochs 3 --e_layer_num 3  --data_seq_len 10 --k 5 --batch_size 64 --mode train \
    --dataset "${DATASET}" --data_path "${DATA_PATH}" --input_c 10 --output_c 10

echo "=== TEST: k=${k}, batch_size=${batch_size} ==="
python main.py --anormly_ratio 1 --win_size 100 --num_epochs 3 --e_layer_num 3  --data_seq_len 10 --k 5 --batch_size 64 --mode test \
    --dataset "${DATASET}" --data_path "${DATA_PATH}" --input_c 10 --output_c 10