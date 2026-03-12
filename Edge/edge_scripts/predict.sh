#!/bin/bash


python threshold_predict.py --score_file prediction_results/Openstack_e6_k1_l3_b96_scores.txt --output_pred prediction_results/Openstack_e6_k1_l3_b96_predictions.txt  --dataset os  --energy_name Openstack_e6_k1_l3_b96  --thresholds_yaml threshold_os.yaml  --label_file dataset/openstack_label.csv

python threshold_predict.py --score_file prediction_results/Openstack_e10_k5_l3_b64_scores.txt --output_pred prediction_results/Openstack_e10_k5_l3_b64_predictions.txt  --dataset os  --energy_name Openstack_e10_k5_l3_b64  --thresholds_yaml threshold_os.yaml  --label_file dataset/openstack_label.csv

python threshold_predict.py --score_file prediction_results/Openstack_e10_k5_l3_b96_scores.txt --output_pred prediction_results/Openstack_e10_k5_l3_b96_predictions.txt  --dataset os  --energy_name Openstack_e10_k5_l3_b96  --thresholds_yaml threshold_os.yaml  --label_file dataset/openstack_label.csv
