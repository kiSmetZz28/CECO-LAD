#!/bin/bash

./cmake-out/executor_runner --model_path ./checkpoints/qbat_hdfs/HDFS_e3_k4_l8_b32.pte --data_path ./dataset/hdfs_processed_data.txt --model_name HDFS_e3_k4_l8_b32 --win_size 50 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_hdfs/HDFS_e3_k5_l6_b96.pte --data_path ./dataset/hdfs_processed_data.txt --model_name HDFS_e3_k5_l6_b96 --win_size 50 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_hdfs/HDFS_e10_k3_l8_b32.pte --data_path ./dataset/hdfs_processed_data.txt --model_name HDFS_e10_k3_l8_b32 --win_size 50 &

wait
