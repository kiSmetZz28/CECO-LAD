#!/bin/bash


python threshold_predict.py --score_file /home/qinxuan/executorch/prediction_results/Openstack_e6_k1_l3_b96_scores.txt --output_pred /home/qinxuan/executorch/prediction_results/Openstack_e6_k1_l3_b96_predictions.txt  --dataset os  --energy_name   Openstack_e6_k1_l3_b96  --thresholds_yaml threshold_os.yaml  --label_file /home/qinxuan/executorch/dataset/OpenStack_label_w100_len10.csv

python threshold_predict.py --score_file /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b64_scores.txt --output_pred /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b64_predictions.txt  --dataset os  --energy_name   Openstack_e10_k5_l3_b64  --thresholds_yaml threshold_os.yaml  --label_file /home/qinxuan/executorch/dataset/OpenStack_label_w100_len10.csv

python threshold_predict.py --score_file /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b96_scores.txt --output_pred /home/qinxuan/executorch/prediction_results/Openstack_e10_k5_l3_b96_predictions.txt  --dataset os  --energy_name   Openstack_e10_k5_l3_b96  --thresholds_yaml threshold_os.yaml  --label_file /home/qinxuan/executorch/dataset/OpenStack_label_w100_len10.csv
