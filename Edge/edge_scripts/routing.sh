#!/bin/bash


python mahalanobis_routing.py 	--test_score_files /home/qinxuan/executorch/prediction_results/Openstack_e6_k1_l3_b96_scores.txt /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b64_scores.txt /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b96_scores.txt\
				--train_score_files /home/qinxuan/executorch/prediction_results/Openstack_e6_k1_l3_b96_train_score.txt /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b64_train_score.txt /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b96_train_score.txt\
				--thresholds_yaml  threshold_os.yaml\
				--dataset  os \
				--energy_names  Openstack_e6_k1_l3_b96 Openstack_e10_k5_l3_b64 Openstack_e10_k5_l3_b96 \
				--output_indices  /home/qinxuan/executorch/prediction_results/os_indices.txt \
				--test_data_file  /home/qinxuan/executorch/dataset/OpenStack_test_data_w100_len10.csv  \
				--output_selected_data  /home/qinxuan/executorch/prediction_results/os_selected.txt
