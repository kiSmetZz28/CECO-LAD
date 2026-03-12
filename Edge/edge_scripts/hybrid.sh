#!/bin/bash


python ensemble_test.py --edge_preds /home/qinxuan/executorch/prediction_results/Openstack_e6_k1_l3_b96_predictions.txt /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b64_predictions.txt /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b96_predictions.txt \
			--label /home/qinxuan/executorch/dataset/OpenStack_label_w100_len10.csv  \
			--cloud_pred /home/qinxuan/executorch/prediction_results/cloud_selected_ensemble_preds.txt   \
			--indices  /home/qinxuan/executorch/prediction_results/os_indices.txt
