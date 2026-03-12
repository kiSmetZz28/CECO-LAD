#!/bin/bash


python ensemble_test.py --edge_preds prediction_results/Openstack_e6_k1_l3_b96_predictions.txt prediction_results/Openstack_e10_k5_l3_b64_predictions.txt prediction_results/Openstack_e10_k5_l3_b96_predictions.txt \
	--label dataset/OpenStack_label_w100_len10.csv \
	--cloud_pred prediction_results/cloud_selected_ensemble_preds.txt \
	--indices prediction_results/os_indices.txt
