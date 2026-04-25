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
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import math
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import resource
import multiprocessing
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
parser.add_argument('--s', default=1, type=int)
parser.add_argument('--jobs', type=int, default=1,
                    help='parallel worker processes for fine matching')
parser.add_argument('--metrics_output', type=str, default='',
                    help='path to save metrics JSON (default: {folder}/metrics_match.json)')
parser.add_argument('--crt_decrypt', action='store_true', help='Use CRT decrypt for DJ')
args = parser.parse_args()

from libs.SV_DJ_cluster.crypto_system import load_private_key, calculate_sim

_private_key = load_private_key(args.key_size, args.s, keys_dir='libs/SV_DJ_cluster/keys')
_crt_factors = None
_center_enrolled = None

if args.crt_decrypt:
    from libs.SV_DJ_cluster.dj_crt_decrypt import load_factors_json
    fac = load_factors_json('libs/SV_DJ_cluster/keys', args.key_size, args.s)
    if fac is not None:
        _crt_factors = (fac[0], fac[1])


def _pool_init_worker(key_size, s, repo_root, crt_enabled, center_enrolled=None):
    """Initialize per-process key material for coarse/fine matching."""
    global _private_key, _crt_factors, _center_enrolled
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    _private_key = load_private_key(key_size, s, keys_dir='libs/SV_DJ_cluster/keys')
    _crt_factors = None
    _center_enrolled = center_enrolled
    if crt_enabled:
        from libs.SV_DJ_cluster.dj_crt_decrypt import load_factors_json
        fac = load_factors_json('libs/SV_DJ_cluster/keys', key_size, s)
        if fac is not None:
            _crt_factors = (fac[0], fac[1])


def _worker_fine_match(task):
    """Compute one fine-stage score for a candidate pair."""
    idx, folder, probe_id, gallery_id, K, L, M = task
    c_probe, C_tilde_probe = load_enrolled_file(os.path.join(folder, '{}.npy'.format(probe_id)))
    c_gal, C_tilde_gal = load_enrolled_file(os.path.join(folder, '{}.npy'.format(gallery_id)))
    score, durations = calculate_sim(
        c_probe, c_gal,
        C_tilde_probe, C_tilde_gal,
        K, L, M, priv=_private_key, crt_tuple=_crt_factors)
    return idx, score, durations[0], durations[1]


def _worker_coarse_probe(task):
    """Compute coarse-stage candidates for one probe."""
    probe_id, folder, K, L, M, n_clusters = task
    start_probe = time.time()
    probe_path = os.path.join(folder, '{}.npy'.format(probe_id))
    c_probe, C_tilde_probe = load_enrolled_file(probe_path)

    center_scores = []
    coarse_plain = 0.0
    coarse_cypher = 0.0
    for c in range(n_clusters):
        c_center, C_tilde_center = _center_enrolled[c]
        score, durations = calculate_sim(
            c_probe, c_center,
            C_tilde_probe, C_tilde_center,
            K, L, M, priv=_private_key, crt_tuple=_crt_factors)
        center_scores.append((c, score))
        coarse_plain += durations[0]
        coarse_cypher += durations[1]

    center_scores.sort(key=lambda x: x[1], reverse=True)
    top_score = center_scores[0][1]
    margin = 0.08          # 允许与最高分相差 0.08 的相似度 0.12
    min_clusters = 10   # 最少强制搜 5 个簇兜底 128-5 256-10 512-20 1024-50
    max_clusters = 25  # 最多搜 25 个簇封顶10%  128-12 256-25 512-50 1024-102

    top_clusters = []
    for i, (cid, score) in enumerate(center_scores):
        if i < min_clusters or (top_score - score <= margin and i < max_clusters):
            top_clusters.append(cid)
        else:
            break

    probe_coarse_time = time.time() - start_probe
    return probe_id, top_clusters, coarse_plain, coarse_cypher, probe_coarse_time

def load_enrolled_file(filepath):
    c_f, C_tilde_f = np.load(filepath, allow_pickle=True)
    return c_f, C_tilde_f


def main():
    L = int(np.ceil(2 ** ((args.key_size * args.s) / (2 * args.K + 9) - 2) - 1))
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
        n_clusters, len(gallery_indices), top_k), flush=True)

    # --- Load encrypted cluster centers into memory ---
    print('[ClusterMatch] Loading encrypted cluster centers...', flush=True)
    centers_dir = os.path.join(folder, 'centers')
    center_enrolled = []
    for c in range(n_clusters):
        c_f, C_tilde_f = load_enrolled_file(
            os.path.join(centers_dir, '{}.npy'.format(c)))
        center_enrolled.append((c_f, C_tilde_f))

    # --- Parse pair list to identify unique probes (lightweight first pass) ---
    print('[ClusterMatch] Scanning pair list...', flush=True)
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
    print('  {} unique probes, {} total pairs'.format(n_probes, n_pairs), flush=True)

    # ============================================================
    # Stage 1: Coarse matching (probe vs cluster centers)
    # ============================================================
    print('[ClusterMatch] Stage 1: Coarse matching ({} probes x {} centers, jobs={})...'.format(
        n_probes, n_clusters, args.jobs), flush=True)
    start_coarse = time.time()
    probe_candidates = {}
    coarse_ops = 0
    coarse_dur_cypher = 0.0
    coarse_dur_plain = 0.0

    per_probe_metrics = {}
    coarse_tasks = [(probe_id, folder, K, L, M, n_clusters) for probe_id in unique_probes]
    if args.jobs > 1:
        chunksize = max(1, len(coarse_tasks) // max(1, args.jobs * 4))
        done = 0
        with multiprocessing.Pool(
            processes=args.jobs,
            initializer=_pool_init_worker,
            initargs=(args.key_size, args.s, _REPO_ROOT, args.crt_decrypt, center_enrolled),
        ) as pool:
            for probe_id, top_cluster_ids, p_plain, p_cypher, probe_coarse_time in pool.imap_unordered(_worker_coarse_probe, coarse_tasks, chunksize=chunksize):
                candidate_gallery = set()
                for cluster_id in top_cluster_ids:
                    candidate_gallery.update(cluster_members[cluster_id])
                probe_candidates[probe_id] = candidate_gallery
                coarse_dur_plain += p_plain
                coarse_dur_cypher += p_cypher
                coarse_ops += n_clusters
                per_probe_metrics[probe_id] = {
                    'coarse_time': probe_coarse_time,
                    'coarse_decrypt_ops': n_clusters,
                    'n_candidates': len(candidate_gallery),
                    'fine_time': 0.0,
                    'fine_decrypt_ops': 0,
                }
                done += 1
                if done % 50 == 0 or done == 1:
                    elapsed = time.time() - start_coarse
                    print('  Coarse: {}/{} probes ({:.1f}s elapsed)'.format(
                        done, n_probes, elapsed), flush=True)
    else:
        _pool_init_worker(args.key_size, args.s, _REPO_ROOT, args.crt_decrypt, center_enrolled)
        for pi, task in enumerate(coarse_tasks):
            probe_id, top_cluster_ids, p_plain, p_cypher, probe_coarse_time = _worker_coarse_probe(task)
            candidate_gallery = set()
            for cluster_id in top_cluster_ids:
                candidate_gallery.update(cluster_members[cluster_id])
            probe_candidates[probe_id] = candidate_gallery
            coarse_dur_plain += p_plain
            coarse_dur_cypher += p_cypher
            coarse_ops += n_clusters
            per_probe_metrics[probe_id] = {
                'coarse_time': probe_coarse_time,
                'coarse_decrypt_ops': n_clusters,
                'n_candidates': len(candidate_gallery),
                'fine_time': 0.0,
                'fine_decrypt_ops': 0,
            }
            if (pi + 1) % 50 == 0 or pi == 0:
                elapsed = time.time() - start_coarse
                print('  Coarse: {}/{} probes ({:.1f}s elapsed)'.format(
                    pi + 1, n_probes, elapsed), flush=True)

    coarse_time = time.time() - start_coarse
    print('  Coarse stage done in {:.2f}s ({} ops)'.format(
        coarse_time, coarse_ops), flush=True)

    # ============================================================
    # Stage 2: Fine matching (stream from pair file, LRU gallery cache)
    # ============================================================
    print('[ClusterMatch] Stage 2: Fine matching (jobs={})...'.format(args.jobs), flush=True)
    start_fine = time.time()
    pair_rows = []
    fine_tasks = []
    genuine_total = 0
    with open(args.pair_list, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) != 3:
                continue
            probe_id = int(parts[0])
            gallery_id = int(parts[1])
            label = int(parts[2])
            is_candidate = gallery_id in probe_candidates.get(probe_id, set())
            row_idx = len(pair_rows)
            pair_rows.append((probe_id, gallery_id, label, is_candidate))
            if is_candidate:
                fine_tasks.append((row_idx, folder, probe_id, gallery_id, K, L, M))
            if label == 1:
                genuine_total += 1

    fine_results = {}
    fine_dur_plain = 0.0
    fine_dur_cypher = 0.0
    if args.jobs > 1 and len(fine_tasks) > 0:
        chunksize = max(1, len(fine_tasks) // max(1, args.jobs * 4))
        with multiprocessing.Pool(
            processes=args.jobs,
            initializer=_pool_init_worker,
            initargs=(args.key_size, args.s, _REPO_ROOT, args.crt_decrypt),
        ) as pool:
            for idx, score, d_plain, d_cypher in pool.imap_unordered(_worker_fine_match, fine_tasks, chunksize=chunksize):
                fine_results[idx] = score
                fine_dur_plain += d_plain
                fine_dur_cypher += d_cypher
                if len(fine_results) % 100000 == 0:
                    elapsed = time.time() - start_fine
                    print('  Fine compute: {}/{} done ({:.1f}s)'.format(
                        len(fine_results), len(fine_tasks), elapsed), flush=True)
    else:
        for task in fine_tasks:
            idx, _folder, probe_id, gallery_id, _K, _L, _M = task
            c_probe, C_tilde_probe = load_enrolled_file(os.path.join(folder, '{}.npy'.format(probe_id)))
            c_gal, C_tilde_gal = load_enrolled_file(os.path.join(folder, '{}.npy'.format(gallery_id)))
            score, durations = calculate_sim(
                c_probe, c_gal,
                C_tilde_probe, C_tilde_gal,
                K, L, M, priv=_private_key, crt_tuple=_crt_factors)
            fine_results[idx] = score
            fine_dur_plain += durations[0]
            fine_dur_cypher += durations[1]

    fine_ops = len(fine_results)
    skipped_ops = len(pair_rows) - fine_ops
    probe_ap_data = defaultdict(lambda: {
        'computed': [],
        'skipped_genuine': 0,
        'skipped_total': 0,
    })
    genuine_hits = 0
    fw = open(args.score_list, 'w')
    for row_idx, (probe_id, gallery_id, label, is_candidate) in enumerate(pair_rows):
        if is_candidate:
            score = fine_results[row_idx]
            per_probe_metrics[probe_id]['fine_decrypt_ops'] += 1
            probe_ap_data[probe_id]['computed'].append((float(score), label))
            if label == 1:
                genuine_hits += 1
        else:
            score = -1
            apd = probe_ap_data[probe_id]
            apd['skipped_total'] += 1
            if label == 1:
                apd['skipped_genuine'] += 1
        fw.write('{} {} {}\n'.format(probe_id, gallery_id, score))
    fw.close()

    # Approximate per-probe fine time by distributing total fine time over per-probe fine ops.
    fine_time_weight = (time.time() - start_fine) / fine_ops if fine_ops > 0 else 0.0
    for pid in unique_probes:
        per_probe_metrics[pid]['fine_time'] = per_probe_metrics[pid]['fine_decrypt_ops'] * fine_time_weight
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

    del probe_ap_data

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
