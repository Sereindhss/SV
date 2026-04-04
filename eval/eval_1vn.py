#!/usr/bin/env python
import argparse
import json
import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.optimize import brentq
from scipy import interpolate

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--pair_list', type=str, help='opensource pair list.')
parser.add_argument('--score_list', type=str, help='opensource score list.')
parser.add_argument('--metrics_output', type=str, default='',
                    help='optional path to save evaluation metrics as JSON')

def perform_1vn_eval(label, scores, metrics_output=''):
    if len(label) == 0 or len(scores) == 0:
        print("Error: No labels or scores found. The score list may be empty.")
        return
    label_arr = np.array(label)
    scores_arr = np.array(scores)

    fpr, tpr, thresholds = roc_curve(label_arr, scores_arr)
    roc_auc = auc(fpr, tpr)

    mAP = float(average_precision_score(label_arr, scores_arr))

    fpr_flip = np.flipud(fpr)
    tpr_flip = np.flipud(tpr)
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    tar_at_far = {}
    to_print = ''
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr_flip - x_labels[fpr_iter]), range(len(fpr_flip)))))
        tar_val = float(tpr_flip[min_index])
        far_key = 'TAR@FAR={}'.format(x_labels[fpr_iter])
        tar_at_far[far_key] = round(tar_val, 6)
        print('  {:0.4f}'.format(tar_val))
        to_print = to_print + '  {:0.4f}'.format(tar_val)
    print(to_print)
    print('  AUC: {:.6f}'.format(roc_auc))
    print('  mAP: {:.6f}'.format(mAP))

    if metrics_output:
        eval_metrics = {
            'n_pairs': int(len(label)),
            'n_genuine': int(np.sum(label_arr == 1)),
            'n_impostor': int(np.sum(label_arr == 0)),
            'AUC': round(roc_auc, 6),
            'mAP': round(mAP, 6),
            'TAR_at_FAR': tar_at_far,
        }
        with open(metrics_output, 'w') as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
        print('[Eval] Metrics saved to {}'.format(metrics_output))

def load_pair_score(pair_list, score_list):
    targets, scores = [], []
    with open(pair_list, 'r') as f_pair, open(score_list, 'r') as f_score:
        for pair_line, score_line in zip(f_pair, f_score):
            parts1 = pair_line.strip().split(' ')
            parts2 = score_line.strip().split(' ')
            assert parts1[0] == parts2[0]
            assert parts1[1] == parts2[1]
            is_same = int(parts1[2])
            score = float(parts2[2])
            targets.append(is_same)
            scores.append(score)
    return targets, scores

def eval(pair_list, score_list, metrics_output=''):
    labels, scores = load_pair_score(pair_list, score_list)
    perform_1vn_eval(labels, scores, metrics_output)

def main():
    args = parser.parse_args()
    eval(args.pair_list, args.score_list, args.metrics_output)

if __name__ == '__main__':
    main()
