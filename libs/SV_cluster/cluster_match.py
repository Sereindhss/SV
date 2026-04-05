#!/usr/bin/env python
"""
Online two-stage encrypted matching for SecureVector with clustering index.

Stage 1 (Coarse): For each unique probe, compute similarity with all C
    cluster centers and select the top-k most similar clusters.
Stage 2 (Fine):   For each pair in the pair list, compute exact SecureVector
    similarity only if the gallery item is in a candidate cluster.
    Otherwise assign score = -1.

Usage:
    python cluster_match.py \
        --folder $FOLD \
        --pair_list $PAIR_LIST \
        --score_list $SCORE_LIST \
        --top_k 5 \
        --key_size 512 --K 64
"""
import sys
import json
import math
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import resource
from collections import defaultdict, OrderedDict
from itertools import repeat

parser = argparse.ArgumentParser(
    description='Two-stage cluster matching in SecureVector')
parser.add_argument('--folder', type=str, required=True,
                    help='folder with enrolled features and cluster index')
parser.add_argument('--pair_list', type=str, required=True,
                    help='pair file')
parser.add_argument('--score_list', type=str, required=True,
                    help='output file for scores')
parser.add_argument('--top_k', type=int, default=5,
                    help='number of candidate clusters per probe')
parser.add_argument('--K', default=64, type=int)
parser.add_argument('--key_size', default=512, type=int)
parser.add_argument('--metrics_output', type=str, default='',
                    help='path to save metrics JSON (default: {folder}/metrics_match.json)')
args = parser.parse_args()

private_key = np.load(
    'libs/SecureVector/keys/privatekey_{}.npy'.format(args.key_size),
    allow_pickle=True)[0]


def load_enrolled_file(filepath):
    c_f, C_tilde_f = np.load(filepath, allow_pickle=True)
    return c_f, C_tilde_f


def decrypt_sum(C_tilde_x, C_tilde_y):
    C_z = private_key.decrypt(C_tilde_x + C_tilde_y)
    return C_z


def decode_uvw(C_f, K, L):
    u_list, v_list = [], []
    for i in range(K):
        next_C_f = C_f // (4 * L)
        u_list.append(C_f - (4 * L) * next_C_f)
        C_f = next_C_f
    for i in range(K):
        next_C_f = C_f // (4 * L)
        v_list.append(C_f - (4 * L) * next_C_f)
        C_f = next_C_f
    w_f = C_f
    u_list.reverse()
    v_list.reverse()
    return u_list, v_list, int(w_f)


def calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M):
    start = time.time()
    C_z = decrypt_sum(C_tilde_x, C_tilde_y)
    duration_cypher = time.time() - start

    start = time.time()
    c_xy = c_x * c_y
    n = len(c_x)
    bar_c_xy = [sum(c_xy[i:i + n // K]) for i in range(0, n, n // K)]

    u_list, v_list, w_z = decode_uvw(C_z, K, L)
    s_list = [1 if v % 2 == 0 else -1 for v in v_list]
    W_z = np.e ** ((w_z - 2**15 * L**8) / (2**14 * L**7 * M))
    score = W_z * sum(
        [bar_c_xy[i] / (s_list[i] * np.e ** ((u_list[i] - 2*L) / M))
         for i in range(K)])
    duration_plain = time.time() - start

    return score, [duration_plain, duration_cypher]


def main():
    L = int(np.ceil(2 ** (args.key_size / (2 * args.K + 9) - 2) - 1))
    M = L / 128
    K = args.K
    top_k = args.top_k
    folder = args.folder

    # --- Load cluster metadata ---
#    meta_path = os.path.join(folder, 'cluster_meta.npz')
#    meta = np.load(meta_path, allow_pickle=True)
#    gallery_indices = meta['gallery_indices']
#    assignments = meta['assignments']
#    n_clusters = int(meta['n_clusters'])

#    if top_k > n_clusters:
#        top_k = n_clusters
#        print('Warning: top_k reduced to {} (total clusters)'.format(top_k))

#    cluster_members = defaultdict(set)
#    for i, gidx in enumerate(gallery_indices):
#        cluster_members[int(assignments[i])].add(int(gidx))
#--tihuan--
    meta_path = os.path.join(folder, 'cluster_meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    gallery_indices = meta['gallery_indices']
    assign_gidx = meta['assign_gidx']
    assign_cid = meta['assign_cid']
    n_clusters = int(meta['n_clusters'])

    if top_k > n_clusters:
        top_k = n_clusters
        print('Warning: top_k reduced to {} (total clusters)'.format(top_k))

    # 构建每个簇的图库成员集合（软聚类下，一个图库ID可能出现在多个簇中）
    cluster_members = defaultdict(set)
    for gidx, cid in zip(assign_gidx, assign_cid):
        cluster_members[int(cid)].add(int(gidx))
#--tihuan--
    print('[ClusterMatch] Index loaded: {} clusters, {} gallery, top_k={}'.format(
        n_clusters, len(gallery_indices), top_k))

    # --- Load encrypted cluster centers into memory ---
    print('[ClusterMatch] Loading encrypted cluster centers...')
    centers_dir = os.path.join(folder, 'centers')
    center_enrolled = []
    for c in range(n_clusters):
        c_f, C_tilde_f = load_enrolled_file(
            os.path.join(centers_dir, '{}.npy'.format(c)))
        center_enrolled.append((c_f, C_tilde_f))

    # --- Parse pair list to identify unique probes (lightweight first pass) ---
    print('[ClusterMatch] Scanning pair list...')
    unique_probes = set()
    n_pairs = 0
    with open(args.pair_list, 'r') as f:
        for line in f:
            n_pairs += 1
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) >= 3:
                unique_probes.add(int(parts[0]))
    unique_probes = sorted(unique_probes)
    n_probes = len(unique_probes)
    print('  {} unique probes, {} total pairs'.format(n_probes, n_pairs))

    # ============================================================
    # Stage 1: Coarse matching (probe vs cluster centers)
    # ============================================================
    print('[ClusterMatch] Stage 1: Coarse matching ({} probes x {} centers)...'.format(
        n_probes, n_clusters))
    start_coarse = time.time()
    probe_candidates = {}
    coarse_ops = 0
    coarse_dur_cypher = 0.0
    coarse_dur_plain = 0.0

    probe_cache = {}
    per_probe_metrics = {}

    for pi, probe_id in enumerate(unique_probes):
        probe_coarse_start = time.time()
        probe_path = os.path.join(folder, '{}.npy'.format(probe_id))
        c_probe, C_tilde_probe = load_enrolled_file(probe_path)
        probe_cache[probe_id] = (c_probe, C_tilde_probe)

        center_scores = []
        probe_coarse_decrypt_ops = 0
        for c in range(n_clusters):
            c_center, C_tilde_center = center_enrolled[c]
            score, durations = calculate_sim(
                c_probe, c_center,
                C_tilde_probe, C_tilde_center,
                K, L, M)
            center_scores.append((c, score))
            coarse_dur_plain += durations[0]
            coarse_dur_cypher += durations[1]
            coarse_ops += 1
            probe_coarse_decrypt_ops += 1

#        center_scores.sort(key=lambda x: x[1], reverse=True)
#        top_clusters = center_scores[:top_k]
#--tihuan2--
        center_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 动态自适应截断 (Adaptive Top-K)
        top_score = center_scores[0][1]
        margin = 0.08  # 允许与最高分相差 0.08 的相似度
        min_clusters = 5   # 最少强制搜 5 个簇兜底
        max_clusters = 25  # 最多搜 25 个簇封顶
        
        top_clusters = []
        for i, (cid, score) in enumerate(center_scores):
            # 只要没超过最大限制，且满足最小数量或在分数边界内，就加入候选
            if i < min_clusters or (top_score - score <= margin and i < max_clusters):
                top_clusters.append((cid, score))
            else:
                break
#--tihuan2--
        candidate_gallery = set()
        for cluster_id, _ in top_clusters:
            candidate_gallery.update(cluster_members[cluster_id])
        probe_candidates[probe_id] = candidate_gallery

        probe_coarse_time = time.time() - probe_coarse_start
        per_probe_metrics[probe_id] = {
            'coarse_time': probe_coarse_time,
            'coarse_decrypt_ops': probe_coarse_decrypt_ops,
            'n_candidates': len(candidate_gallery),
            'fine_time': 0.0,
            'fine_decrypt_ops': 0,
        }

        if (pi + 1) % 50 == 0 or pi == 0:
            elapsed = time.time() - start_coarse
            print('  Coarse: {}/{} probes ({:.1f}s elapsed)'.format(
                pi + 1, n_probes, elapsed))

    coarse_time = time.time() - start_coarse
    print('  Coarse stage done in {:.2f}s ({} ops)'.format(
        coarse_time, coarse_ops))

    # ============================================================
    # Stage 2: Fine matching (stream from pair file, LRU gallery cache)
    # ============================================================
    print('[ClusterMatch] Stage 2: Fine matching...')
    start_fine = time.time()
    fw = open(args.score_list, 'w')

    fine_ops = 0
    skipped_ops = 0
    genuine_hits = 0
    genuine_total = 0
    fine_dur_cypher = 0.0
    fine_dur_plain = 0.0

    GALLERY_CACHE_MAX = 2000
    gallery_cache = OrderedDict()

    probe_ap_data = defaultdict(lambda: {
        'computed': [],
        'skipped_genuine': 0,
        'skipped_total': 0,
    })

    line_idx = 0
    with open(args.pair_list, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                line_idx += 1
                continue
            parts = line.split(' ')
            if len(parts) != 3:
                line_idx += 1
                continue

            probe_id = int(parts[0])
            gallery_id = int(parts[1])
            label = int(parts[2])

            if label == 1:
                genuine_total += 1

            candidates = probe_candidates.get(probe_id, set())

            if gallery_id in candidates:
                fine_start_single = time.time()

                if probe_id in probe_cache:
                    c_probe, C_tilde_probe = probe_cache[probe_id]
                else:
                    c_probe, C_tilde_probe = load_enrolled_file(
                        os.path.join(folder, '{}.npy'.format(probe_id)))

                if gallery_id in gallery_cache:
                    gallery_cache.move_to_end(gallery_id)
                    c_gal, C_tilde_gal = gallery_cache[gallery_id]
                else:
                    c_gal, C_tilde_gal = load_enrolled_file(
                        os.path.join(folder, '{}.npy'.format(gallery_id)))
                    if len(gallery_cache) >= GALLERY_CACHE_MAX:
                        gallery_cache.popitem(last=False)
                    gallery_cache[gallery_id] = (c_gal, C_tilde_gal)

                score, durations = calculate_sim(
                    c_probe, c_gal,
                    C_tilde_probe, C_tilde_gal,
                    K, L, M)
                fine_dur_plain += durations[0]
                fine_dur_cypher += durations[1]
                fine_ops += 1

                fine_elapsed_single = time.time() - fine_start_single
                per_probe_metrics[probe_id]['fine_time'] += fine_elapsed_single
                per_probe_metrics[probe_id]['fine_decrypt_ops'] += 1

                if label == 1:
                    genuine_hits += 1

                probe_ap_data[probe_id]['computed'].append(
                    (float(score), label))
            else:
                score = -1
                skipped_ops += 1
                apd = probe_ap_data[probe_id]
                apd['skipped_total'] += 1
                if label == 1:
                    apd['skipped_genuine'] += 1

            fw.write('{} {} {}\n'.format(probe_id, gallery_id, score))

            line_idx += 1
            if line_idx % 100000 == 0:
                elapsed = time.time() - start_fine
                print('  Fine: {}/{} pairs ({} computed, {} skipped, {:.1f}s)'.format(
                    line_idx, n_pairs,
                    fine_ops, skipped_ops, elapsed))

    fw.close()
    fine_time = time.time() - start_fine
    total_time = coarse_time + fine_time

    # --- Compute per-query stats ---
    n_gallery_total = len(gallery_indices)
    linear_scan_ops = n_probes * n_gallery_total
    cluster_ops = coarse_ops + fine_ops
    recall = genuine_hits / genuine_total * 100 if genuine_total > 0 else 0
    speedup = linear_scan_ops / cluster_ops if cluster_ops > 0 else float('inf')

    query_total_times = []
    query_decrypt_ops_list = []
    for pid in unique_probes:
        pm = per_probe_metrics[pid]
        qt = pm['coarse_time'] + pm['fine_time']
        qd = pm['coarse_decrypt_ops'] + pm['fine_decrypt_ops']
        query_total_times.append(qt)
        query_decrypt_ops_list.append(qd)

    qt_arr = np.array(query_total_times)
    qd_arr = np.array(query_decrypt_ops_list)

    # --- Compute mAP (memory-efficient) ---
    ap_list = []
    for pid in unique_probes:
        apd = probe_ap_data.get(pid)
        if apd is None:
            continue
        computed = apd['computed']
        n_skipped_genuine = apd['skipped_genuine']
        n_skipped_total = apd['skipped_total']
        total_genuine = sum(1 for _, l in computed if l == 1) + n_skipped_genuine
        if total_genuine == 0:
            continue
        computed_sorted = sorted(computed, key=lambda x: x[0], reverse=True)
        n_relevant = 0
        cum_precision_sum = 0.0
        for rank, (sc, lbl) in enumerate(computed_sorted, 1):
            if lbl == 1:
                n_relevant += 1
                cum_precision_sum += n_relevant / rank
        n_computed = len(computed_sorted)
        for j in range(n_skipped_genuine):
            n_relevant += 1
            rank = n_computed + n_skipped_total - n_skipped_genuine + j + 1
            cum_precision_sum += n_relevant / rank
        if n_relevant > 0:
            ap_list.append(cum_precision_sum / n_relevant)
    mAP = float(np.mean(ap_list)) if ap_list else 0.0

    del probe_cache, gallery_cache, probe_ap_data

    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    cand_sizes = [len(v) for v in probe_candidates.values()]
    avg_cand = np.mean(cand_sizes)

    print('\n[ClusterMatch] ====== Results ======')
    print('  Clusters (C): {}'.format(n_clusters))
    print('  Top-k: {}'.format(top_k))
    print('  Avg candidates per probe: {:.0f} / {} gallery'.format(
        avg_cand, n_gallery_total))
    print('')
    print('[Metrics] Coarse stage time: {:.4f}s'.format(coarse_time))
    print('[Metrics] Fine stage time: {:.4f}s'.format(fine_time))
    print('[Metrics] Total matching time: {:.4f}s'.format(total_time))
    print('[Metrics] Per-query time: avg={:.4f}s, median={:.4f}s, '
          'min={:.4f}s, max={:.4f}s'.format(
              qt_arr.mean(), np.median(qt_arr), qt_arr.min(), qt_arr.max()))
    print('[Metrics] Coarse decrypt ops: {}'.format(coarse_ops))
    print('[Metrics] Fine decrypt ops: {}'.format(fine_ops))
    print('[Metrics] Total decrypt ops: {} (linear would be {})'.format(
        cluster_ops, linear_scan_ops))
    print('[Metrics] Decrypt ops speedup: {:.1f}x'.format(speedup))
    print('[Metrics] Avg decrypt ops per query: {:.1f}'.format(qd_arr.mean()))
    print('[Metrics] Genuine recall: {}/{} = {:.2f}%'.format(
        genuine_hits, genuine_total, recall))
    print('[Metrics] mAP: {:.4f}'.format(mAP))
    print('[Metrics] Paillier time (coarse): {:.4f}s'.format(coarse_dur_cypher))
    print('[Metrics] Paillier time (fine): {:.4f}s'.format(fine_dur_cypher))
    print('[Metrics] Peak RAM: {:.2f} MB'.format(peak_mem))

    # --- Save metrics JSON ---
    metrics = {
        'params': {
            'n_clusters': n_clusters,
            'top_k': top_k,
            'key_size': args.key_size,
            'K': K,
        },
        'dataset': {
            'gallery_size': int(n_gallery_total),
            'n_probes': int(n_probes),
            'n_pairs': int(n_pairs),
        },
        'efficiency': {
            'coarse_time_s': round(coarse_time, 4),
            'fine_time_s': round(fine_time, 4),
            'total_time_s': round(total_time, 4),
            'per_query_avg_s': round(float(qt_arr.mean()), 6),
            'per_query_median_s': round(float(np.median(qt_arr)), 6),
            'per_query_min_s': round(float(qt_arr.min()), 6),
            'per_query_max_s': round(float(qt_arr.max()), 6),
            'coarse_decrypt_ops': int(coarse_ops),
            'fine_decrypt_ops': int(fine_ops),
            'total_decrypt_ops': int(cluster_ops),
            'linear_scan_ops': int(linear_scan_ops),
            'speedup_ratio': round(speedup, 2),
            'avg_decrypt_ops_per_query': round(float(qd_arr.mean()), 2),
            'paillier_time_coarse_s': round(coarse_dur_cypher, 4),
            'paillier_time_fine_s': round(fine_dur_cypher, 4),
            'paillier_time_total_s': round(coarse_dur_cypher + fine_dur_cypher, 4),
            'plain_time_coarse_s': round(coarse_dur_plain, 4),
            'plain_time_fine_s': round(fine_dur_plain, 4),
        },
        'accuracy': {
            'genuine_recall_pct': round(recall, 4),
            'genuine_hits': int(genuine_hits),
            'genuine_total': int(genuine_total),
            'mAP': round(mAP, 6),
        },
        'overhead': {
            'avg_candidates_per_probe': round(float(avg_cand), 2),
            'peak_ram_mb': round(peak_mem, 2),
        },
        'time_complexity_datapoint': {
            'N_gallery': int(n_gallery_total),
            'avg_query_time_s': round(float(qt_arr.mean()), 6),
            'avg_query_decrypt_ops': round(float(qd_arr.mean()), 2),
        },
        'per_probe_detail': [
            {
                'probe_id': int(pid),
                'coarse_time_s': round(per_probe_metrics[pid]['coarse_time'], 6),
                'fine_time_s': round(per_probe_metrics[pid]['fine_time'], 6),
                'total_time_s': round(
                    per_probe_metrics[pid]['coarse_time'] +
                    per_probe_metrics[pid]['fine_time'], 6),
                'coarse_decrypt_ops': per_probe_metrics[pid]['coarse_decrypt_ops'],
                'fine_decrypt_ops': per_probe_metrics[pid]['fine_decrypt_ops'],
                'n_candidates': per_probe_metrics[pid]['n_candidates'],
            }
            for pid in unique_probes
        ],
    }

    metrics_path = args.metrics_output
    if not metrics_path:
        metrics_path = os.path.join(folder, 'metrics_match.json')
    with open(metrics_path, 'w') as mf:
        json.dump(metrics, mf, indent=2, ensure_ascii=False)
    print('[ClusterMatch] Metrics saved to {}'.format(metrics_path))


if __name__ == '__main__':
    main()
