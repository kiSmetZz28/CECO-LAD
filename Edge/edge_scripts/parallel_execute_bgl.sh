#!/bin/bash

./cmake-out/executor_runner --model_path ./checkpoints/qbat_bgl/BGL_e3_k4_l3_b32.pte --data_path ./dataset/bgl_processed_data.txt --model_name BGL_e3_k4_l3_b32 --win_size 100 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_bgl/BGL_e3_k5_l3_b96.pte --data_path ./dataset/bgl_processed_data.txt --model_name BGL_e3_k5_l3_b96 --win_size 100 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_bgl/BGL_e6_k4_l3_b64.pte --data_path ./dataset/bgl_processed_data.txt --model_name BGL_e6_k4_l3_b64 --win_size 100 &

wait
