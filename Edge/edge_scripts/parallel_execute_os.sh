#!/bin/bash

./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e6_k1_l3_b96.pte --data_path ./dataset/os_test.txt --mode train --model_name Openstack_e6_k1_l3_b96 --win_size 100 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e10_k5_l3_b64.pte --data_path ./dataset/os_test.txt  --mode train --model_name Openstack_e10_k5_l3_b64 --win_size 100 &
./cmake-out/executor_runner --model_path ./checkpoints/qbat_os/Openstack_e10_k5_l3_b96.pte --data_path ./dataset/os_test.txt  --mode train --model_name Openstack_e10_k5_l3_b96 --win_size 100 &

wait
