#!/bin/bash


python em_gmm_threshold.py --train_energy prediction_results/Openstack_e6_k1_l3_b96_train_score.txt  --dataset os  --energy_name Openstack_e6_k1_l3_b96  --yaml_output threshold_os.yaml

python em_gmm_threshold.py --train_energy prediction_results/Openstack_e10_k5_l3_b64_train_score.txt  --dataset os  --energy_name Openstack_e10_k5_l3_b64  --yaml_output threshold_os.yaml

python em_gmm_threshold.py --train_energy prediction_results/Openstack_e10_k5_l3_b96_train_score.txt  --dataset os  --energy_name Openstack_e10_k5_l3_b96  --yaml_output threshold_os.yaml
