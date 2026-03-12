#!/bin/bash

# Train mode runs the models on the
# training set to produce anomaly scores used later for
# threshold estimation (EM-GMM in em_gmm_threshold.py).
#
# ./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e6_k1_l3_b96.pte --data_path ./dataset/os_processed_train_data.txt --mode train --model_name Openstack_e6_k1_l3_b96 --win_size 100 &
# ./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e10_k5_l3_b64.pte --data_path ./dataset/os_processed_train_data.txt  --mode train --model_name Openstack_e10_k5_l3_b64 --win_size 100 &
# ./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e10_k5_l3_b96.pte --data_path ./dataset/os_processed_train_data.txt  --mode train --model_name Openstack_e10_k5_l3_b96 --win_size 100 &

# Test mode runs the models on the test set
# to produce anomaly scores used for prediction, routing
# and the final edge/hybrid evaluation.
./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e6_k1_l3_b96.pte --data_path ./dataset/os_processed_data.txt --mode test --model_name Openstack_e6_k1_l3_b96 --win_size 100 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e10_k5_l3_b64.pte --data_path ./dataset/os_processed_data.txt  --mode test --model_name Openstack_e10_k5_l3_b64 --win_size 100 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e10_k5_l3_b96.pte --data_path ./dataset/os_processed_data.txt  --mode test --model_name Openstack_e10_k5_l3_b96 --win_size 100 &

wait
