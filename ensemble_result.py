import datetime
import os
import argparse
from sklearn.metrics import f1_score

# Import torch after importing and using portable_lib to demonstrate that
# portable_lib works without importing this first.
import torch
import torch.nn as nn
import numpy as np
import time
import yaml
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import logging
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from itertools import combinations


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def ensemble_method(method, ensemble_data):
    if method == 'majority':  # 1. majority voting
        return (ensemble_data.sum(axis=1) >= (ensemble_data.shape[1] // 2 + 1)).astype(int)
    elif method == 'at least one':  # 2. at least one voting
        return np.any(ensemble_data == 1, axis=1).astype(int)
    elif method == 'consensus':  # 3. consensus voting
        return np.all(ensemble_data == 1, axis=1).astype(int)
    return None


"""
    Return indices where model predictions disagree.
"""


def get_disagreement_indices(all_preds):
    disagreement_mask = np.any(all_preds != all_preds[:, [0]], axis=1)
    return np.where(disagreement_mask)[0].tolist()


"""
    Return indices where model predictions disagree.
"""


def get_abnormal_indices(all_preds):
    abnormal_indices = set()

    for preds in all_preds.T:
        abnormal_indices.update(np.where(preds == 1)[0])

    return sorted(abnormal_indices)


"""
    Return indices where any model's score to the threshold point is in the lowest top t (tolerance).
"""


def get_threshold_indices_by_distance(all_scores, thresholds, type, tolerance=0.1):
    threshold_point = np.array(thresholds)

    samples = all_scores

    if type == 'ma':
        cov_matrix = np.cov(samples, rowvar=False)
        inv_covmat = np.linalg.inv(cov_matrix)

    all_distances = []

    for i, sample in enumerate(samples):
        if type == 'eu':
            distance = euclidean(sample, threshold_point)
        elif type == 'ma':
            distance = mahalanobis(sample, threshold_point, inv_covmat)
        else:
            raise ValueError("Invalid distance type. Use 'eu' or 'ma'.")
        all_distances.append((i, distance))  # store index and distance

    # Sort by distance
    all_distances.sort(key=lambda x: x[1], reverse=True)
    num_select = int(len(all_distances) * tolerance)

    # Get the indices of the lowest distances
    selected_indices = [idx for idx, _ in all_distances[:num_select]]

    return sorted(selected_indices)


"""
    Return the union of disagreement and soft-threshold indices. As long as appear in one of them, send to cloud
"""


def get_combined_indices(disagreement_indices, soft_threshold_indices):
    return sorted(set(disagreement_indices).union(set(soft_threshold_indices)))


def performance(gt, pred):
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    logging.info("Accuracy : {:0.6f}, Precision : {:0.6f}, Recall : {:0.6f}, F-score : {:0.6f} ".format(
        accuracy, precision, recall, f_score))
    return f_score


if __name__ == "__main__":

    # Set up logging to file
    log_filename = f'Hybrid_results_bgl_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Optional: keeps logging.infoing to console too
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ensemble_config_bgl.yaml')
    parser.add_argument('--decision_choice', type=str, default='ma')  # voting, eu, ma, all-abnormal
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    filename = cfg['data']['filename']
    label = cfg['data']['label']
    os.makedirs(cfg['data']['output_dir'], exist_ok=True)

    logging.info(f'Loading data from {label}')

    all_scores = []
    all_preds = []
    thresholds = []

    # load labels
    labels = np.loadtxt(label, dtype=int)
    labels = labels.reshape(-1)
    
    model_names = [model_cfg['name'] for model_cfg in cfg['models']]

    # prediction each model
    for model_cfg in cfg['models']:
        logging.info(f"Running inference for {model_cfg['name']}...")
        thresholds.append(model_cfg['threshold'])

        scores = np.loadtxt(os.path.join(cfg['data']['output_dir'], f"{model_cfg['name']}_scores.txt"), dtype=float)
        scores = scores.reshape(-1)
        
        preds = np.loadtxt(os.path.join(cfg['data']['output_dir'], f"{model_cfg['name']}_predictions.txt"), dtype=int)
        preds = preds.reshape(-1)

        # scores = np.loadtxt(f"/home/qinxuan.shi/Desktop/Ensemble/hdfs_energy/test_energy_HDFS_{model_cfg['name']}.txt", dtype=float)
        # scores = scores.reshape(-1)

        # preds = np.loadtxt(f"/home/qinxuan.shi/Desktop/Ensemble/pred_results/hdfs_preds/{model_cfg['name']}.csv", dtype=int)
        # preds = preds.reshape(-1)

        all_scores.append(scores)
        all_preds.append(preds)

    all_preds = np.array(all_preds).T
    all_scores = np.array(all_scores).T

    f1_scores = [f1_score(labels, all_preds[:, i]) for i in range(all_preds.shape[1])]
    
    # Sort models by F1-score in descending order
    sorted_indices = np.argsort(f1_scores)[::-1]
    
    all_preds = all_preds[:, sorted_indices]
    all_scores = all_scores[:, sorted_indices]
    thresholds = [thresholds[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    for name, f1 in zip(model_names, f1_scores):
        logging.info(f"{name}: F1 = {f1:.4f}")

    # target_f1_min = 0.991
    # target_f1_max = 0.992
    
    # all_combines = []

    # for model_idxs in combinations(range(len(model_names)), 3):
    #     selected_preds = all_preds[:, model_idxs]
    
    #     for method in ['majority']:
    #         logging.info(f"===========Edge ensemble result with {method} voting ==========")
    #         ensemble_result = ensemble_method(method, selected_preds)
    #         f1 = performance(labels, ensemble_result)
    
    #         # Check lower and upper bound
    #         if target_f1_min < f1 < target_f1_max:
    #             logging.info(f"===========The three models combination are {model_idxs}==========")
    #             name = [model_names[i] for i in model_idxs]
    #             all_combines.append(name)
    #             logging.info(f"===========The three models combination are {name}==========")

    # 'Openstack_e3_k1_l8_b64', 'Openstack_e6_k5_l3_b32', 'Openstack_e10_k1_l3_b96' 98.61
    # 'Openstack_e3_k1_l6_b64', 'Openstack_e6_k3_l6_b96', 'Openstack_e10_k1_l6_b32' 98.49
    # 'Openstack_e3_k1_l6_b64', 'Openstack_e6_k5_l3_b32', 'Openstack_e10_k1_l3_b96' 98.64
    # 'Openstack_e3_k1_l8_b64', 'Openstack_e6_k3_l8_b96', 'Openstack_e10_k1_l6_b96' 98.58
    # 'Openstack_e3_k1_l8_b64', 'Openstack_e6_k1_l8_b64', 'Openstack_e6_k5_l6_b64' 98.60
    # 'Openstack_e3_k1_l8_b64', 'Openstack_e6_k1_l3_b96', 'Openstack_e10_k3_l3_b64' 98.58
    # 'Openstack_e3_k1_l8_b64', 'Openstack_e6_k1_l3_b96', 'Openstack_e10_k1_l3_b64' 98.74

    # 'e3_k4_l3_b32', 'e3_k5_l3_b96', 'e6_k5_l3_b64'   bgl    99.10
    # 'e3_k4_l3_b32', 'e3_k5_l3_b96', 'e6_k4_l3_b64'   bgl    99.12
    # 'Openstack_e10_k5_l3_b96', 'Openstack_e10_k5_l3_b64', 'Openstack_e6_k1_l3_b96'   os   99.03
    # 'HDFS_e3_k4_l8_b32', 'HDFS_e3_k5_l6_b96', 'HDFS_e10_k3_l3_b32'

    # print(len(all_combines))

    # for combin in all_combines:

    chosen_models = ['e3_k4_l3_b32', 'e3_k5_l3_b96', 'e6_k4_l3_b64']
    # chosen_models = combin
    chosen_indices = [model_names.index(name) for name in chosen_models]
    chosen_preds = all_preds[:, chosen_indices]
    chosen_scores = all_scores[:, chosen_indices]
    chosen_thresholds = [thresholds[i] for i in chosen_indices]

    # unique_combinations, counts = np.unique(chosen_preds, axis=0, return_counts=True)
    # for combo, count in zip(unique_combinations, counts):
    #     print(f"Combo: {combo}, Count: {count}")

    # ensemble result on edge
    logging.info(f"\n===========Edge ensemble result with majority voting ==========")
    ensemble_results_mj = ensemble_method('majority', chosen_preds)
    performance(labels, ensemble_results_mj)

    logging.info(f"===========Edge ensemble result with consensus voting ==========")
    ensemble_results_co = ensemble_method('consensus', chosen_preds)
    performance(labels, ensemble_results_co)

    logging.info(f"===========Edge ensemble result with at least one voting ==========")
    ensemble_results_alo = ensemble_method('at least one', chosen_preds)
    performance(labels, ensemble_results_alo)

    # Read input data again
    with open(filename, 'r') as f:
        input_lines = [line.strip() for line in f.readlines()]

    if args.decision_choice == 'voting':
        # 1. get the indices that predictions are not the same: voting
        selected_indices = get_disagreement_indices(chosen_preds)
    elif args.decision_choice == 'all-abnormal':
        # 2. get all the abnormal predicitons
        selected_indices = get_abnormal_indices(chosen_preds)
    else:
        # 3. get the indices that the anomaly scores are close to the threshold point (3d) ['eu distance' and 'ma distance'].
        selected_indices = get_threshold_indices_by_distance(chosen_scores, chosen_thresholds, args.decision_choice,
                                                            tolerance=0.1)

    logging.info(f"==========={len(selected_indices)} are selected to transmit to cloud==========")
    # combine the indices together
    # combined_indices = get_combined_indices(disagreement_indices, soft_threshold_indices)

    # use the prediction result on cloud to replace the selected indices place.
    # cloud_preds = np.loadtxt('./ensemble_results/ensemble_cloud_pred_result_hdfs.csv', dtype=int)
    cloud_preds = np.loadtxt('label_data_bglw100b64l3.txt', dtype=int).reshape(-1)
    logging.info(f"\n=========== Performance of cloud predictions==========")
    # cloud_ensemble = ensemble_method('majority', cloud_preds)
    performance(labels, cloud_preds)

    logging.info(f"\n===========Hybrid result using {args.decision_choice} policy==========\n")

    # for each ensemble method, replace the selected indices with the cloud results => the hybrid model result
    logging.info(f"\n===========Hybrid result cloud-edge ==========")
    ensemble_results_mj[selected_indices] = cloud_preds[selected_indices]
    performance(labels, ensemble_results_mj)

    # logging.info(f"\n===========Hybrid result using consensus voting at edge==========")
    # ensemble_results_co[selected_indices] = cloud_ensemble[selected_indices]
    # performance(labels, ensemble_results_co)
    #
    # logging.info(f"\n===========Hybrid result using at least one voting at edge==========")
    # ensemble_results_alo[selected_indices] = cloud_ensemble[selected_indices]
    # performance(labels, ensemble_results_alo)
