#!/usr/bin/env python
"""
Offline clustering index builder for SecureVector.
Clusters gallery features using K-Means, encrypts cluster centers
via SecureVector enrollment, and saves the two-level index structure.

Usage:
    python build_index.py \
        --feat_list $FEAT_LIST \
        --pair_list $PAIR_LIST \
        --folder $FOLD \
        --n_clusters 100 \
        --key_size 512 --K 64
"""
import sys
import json
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import resource
import faiss
from itertools import repeat
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='Build clustering index for SecureVector')
parser.add_argument('--feat_list', type=str, required=True,
                    help='path to template feature list')
parser.add_argument('--pair_list', type=str, required=True,
                    help='path to pair list (used to identify gallery indices)')
parser.add_argument('--folder', type=str, required=True,
                    help='folder with enrolled features; cluster index is added here')
parser.add_argument('--n_clusters', type=int, default=100,
                    help='number of K-Means clusters (C)')
parser.add_argument('--K', default=64, type=int)
parser.add_argument('--key_size', default=512, type=int)
parser.add_argument('--public_key', default='libs/SecureVector/keys/publickey',
                    type=str)
parser.add_argument('--metrics_output', type=str, default='',
                    help='path to save metrics JSON (default: {folder}/metrics_build.json)')
args = parser.parse_args()


def load_features(feature_list):
    features = []
    with open(feature_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        feature = [float(e) for e in parts[1:]]
        feature = feature / np.linalg.norm(np.array(feature))
        features.append(feature)
    return np.array(features)


def enroll(feature, K, L, M, public_key):
    start = time.time()
    u_list = [int(e) for e in np.random.rand(K) * (2 * L)]
    v_list = [int(e) for e in np.random.rand(K) * (2 * L)]
    s_list = [1 if v % 2 == 0 else -1 for v in v_list]

    n = len(feature)
    scale = [s_list[i] * np.e ** ((u_list[i] - L) / M) for i in range(K)]
    b_f = [x for item in scale for x in repeat(item, n // K)] * feature
    W_f = np.linalg.norm(b_f)
    c_f = b_f / W_f

    base = [(4 * L) ** (K - 1 - i) for i in range(K)]
    w_f = int((np.log(W_f) + L / M) / (2 * L / M) * 2 ** 15 * L ** 8)
    C_f = np.dot(u_list, base) + \
        np.dot(v_list, base) * (4 * L) ** (K) + \
        w_f * (4 * L) ** (2 * K)
    duration_plain = time.time() - start

    start = time.time()
    C_tilde_f = public_key.encrypt(C_f)
    duration_cypher = time.time() - start

    return [c_f, C_tilde_f], [duration_plain, duration_cypher]


def extract_gallery_indices(pair_list):
    """Extract unique gallery indices (column 2) from the pair list."""
    gallery_set = set()
    with open(pair_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) >= 3:
                gallery_set.add(int(parts[1]))
    return sorted(gallery_set)


def main():
    L = int(np.ceil(2 ** (args.key_size / (2 * args.K + 9) - 2) - 1))
    M = L / 128
    K = args.K
    n_clusters = args.n_clusters

    print('[ClusterIndex] Loading features...')
    features = load_features(args.feat_list)
    n_total = len(features)

    print('[ClusterIndex] Extracting gallery indices from pair list...')
    gallery_indices = np.array(extract_gallery_indices(args.pair_list))
    n_gallery = len(gallery_indices)
    print('  Total features: {}, Gallery features: {}'.format(n_total, n_gallery))

    if n_clusters > n_gallery:
        n_clusters = n_gallery
        print('  Warning: n_clusters reduced to {} (gallery size)'.format(n_clusters))

    gallery_feats = features[gallery_indices]

#    print('[ClusterIndex] Running K-Means clustering (C={})...'.format(n_clusters))
#    start = time.time()
#    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
#    kmeans.fit(gallery_feats)
#    cluster_duration = time.time() - start
#    print('  K-Means completed in {:.2f}s'.format(cluster_duration))

#    centers = kmeans.cluster_centers_
#    center_norms = np.linalg.norm(centers, axis=1)
#    centers = centers / center_norms[:, np.newaxis]
#    assignments = kmeans.labels_

#    for c in range(n_clusters):
#        members = np.sum(assignments == c)
#        if members == 0:
#            print('  Warning: cluster {} has 0 members'.format(c))

#    print('[ClusterIndex] Cluster size stats: min={}, max={}, mean={:.1f}'.format(
#        min(np.bincount(assignments)), max(np.bincount(assignments)),
#        np.mean(np.bincount(assignments))))

#--tihuan--
#    centers = kmeans.cluster_centers_
#    center_norms = np.linalg.norm(centers, axis=1)

    print('[ClusterIndex] Running Spherical K-Means via Faiss (C={})...'.format(n_clusters))
    start = time.time()
    
    d = gallery_feats.shape[1]
    # Faiss 严格要求数据类型为 float32
    gallery_feats_f32 = gallery_feats.astype(np.float32)
    
    # spherical=True 是关键：强制在每次迭代中对中心进行 L2 归一化，完美对齐余弦空间
    kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, verbose=False, spherical=True, seed=42)
    kmeans.train(gallery_feats_f32)
    
    centers = kmeans.centroids
    
    print('  [ClusterIndex] Running Soft Assignment (软聚类分配 - 余弦相似度版本)...')
    # 利用 faiss 构建内积（余弦相似度）索引，用于快速计算特征到中心的得分
    index = faiss.IndexFlatIP(d)
    index.add(centers)
    
    # 搜索每个图库特征距离最近的 2 个簇中心
    # scores 返回的是内积相似度（越大越近）
    scores, sorted_center_indices = index.search(gallery_feats_f32, 2)
    cluster_duration = time.time() - start
    print('  Spherical K-Means completed in {:.2f}s'.format(cluster_duration))
    
    # 软聚类阈值：第二名的相似度必须达到第一名的 85% 以上才算边界样本
    threshold_ratio = 0.70  
    
    assign_gidx = []
    assign_cid = []
    
    for i in range(n_gallery):
        c1 = sorted_center_indices[i, 0] # 最相似的簇
        c2 = sorted_center_indices[i, 1] # 第二相似的簇
        
        s1 = scores[i, 0]
        s2 = scores[i, 1]
        
        # 无条件加入最相似的簇
        assign_gidx.append(gallery_indices[i])
        assign_cid.append(c1)
        
        # 边界样本判断：内积相似度越大越好，比值 > threshold_ratio
        if s2 / (s1 + 1e-9) > threshold_ratio:
            assign_gidx.append(gallery_indices[i])
            assign_cid.append(c2)
            
    assign_gidx = np.array(assign_gidx)
    assign_cid = np.array(assign_cid)
    
    # 兼容后面的指标统计
    cluster_sizes = np.bincount(assign_cid, minlength=n_clusters)

    for c in range(n_clusters):
        if cluster_sizes[c] == 0:
            print('  Warning: cluster {} has 0 members'.format(c))

    print('[ClusterIndex] Cluster size stats: min={}, max={}, mean={:.1f}'.format(
        min(cluster_sizes), max(cluster_sizes), np.mean(cluster_sizes)))
    print('  [ClusterIndex] Soft Assignment expanded total index items from {} to {}'.format(
        n_gallery, len(assign_gidx)))
        
    # 由于 faiss 的 centroids 已经归一化了，这里直接提取范数全为1，兼容原代码写入 meta 的结构
    center_norms = np.linalg.norm(centers, axis=1)


#--tihuan--

    centers_dir = os.path.join(args.folder, 'centers')
    os.makedirs(centers_dir, exist_ok=True)

    print('[ClusterIndex] Encrypting {} cluster centers...'.format(n_clusters))
    pubkey_path = '{}_{}.npy'.format(args.public_key, args.key_size)
    publickey = np.load(pubkey_path, allow_pickle=True)[0]

    start = time.time()
    for c in range(n_clusters):
        result, durations = enroll(centers[c], K, L, M, publickey)
        np.save(os.path.join(centers_dir, '{}.npy'.format(c)),
                np.array(result, np.dtype(object)))
        if (c + 1) % 10 == 0 or c == 0:
            print('  {}/{} centers encrypted'.format(c + 1, n_clusters))
    enroll_duration = time.time() - start
    print('  Center encryption completed in {:.2f}s'.format(enroll_duration))

#    meta_path = os.path.join(args.folder, 'cluster_meta.npz')
#    np.savez(meta_path,
#             gallery_indices=gallery_indices,
#             assignments=assignments,
#             n_clusters=np.array(n_clusters),
#             center_norms=center_norms)
    meta_path = os.path.join(args.folder, 'cluster_meta.npz')
    np.savez(meta_path,
             gallery_indices=gallery_indices,
             assign_gidx=assign_gidx,      # 替换为软聚类的映射
             assign_cid=assign_cid,        # 替换为软聚类的映射
             n_clusters=np.array(n_clusters),
             center_norms=center_norms)
    print('[ClusterIndex] Saved cluster metadata to {}'.format(meta_path))

    total_build_time = cluster_duration + enroll_duration

    # --- Storage overhead ---
    storage_centers = sum(
        os.path.getsize(os.path.join(centers_dir, f))
        for f in os.listdir(centers_dir) if f.endswith('.npy'))
    storage_meta = os.path.getsize(meta_path)
    storage_index_total = storage_centers + storage_meta

    sample_ids = gallery_indices[:min(10, n_gallery)]
    sample_sizes = []
    for sid in sample_ids:
        p = os.path.join(args.folder, '{}.npy'.format(sid))
        if os.path.exists(p):
            sample_sizes.append(os.path.getsize(p))
    avg_enrolled_size = np.mean(sample_sizes) if sample_sizes else 0
    storage_gallery_estimate = int(avg_enrolled_size * n_gallery)
    storage_inflation = (storage_index_total / storage_gallery_estimate
                         if storage_gallery_estimate > 0 else 0)
    assignments = assign_cid
    cluster_sizes = np.bincount(assignments)
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print('\n[ClusterIndex] Summary:')
    print('  Clusters (C): {}'.format(n_clusters))
    print('  Gallery size (N): {}'.format(n_gallery))
    print('  Avg cluster size (N/C): {:.1f}'.format(n_gallery / n_clusters))
    print('  K-Means time: {:.2f}s'.format(cluster_duration))
    print('  Center encryption time: {:.2f}s'.format(enroll_duration))
    print('  Total build time: {:.2f}s'.format(total_build_time))
    print('  Storage - centers: {} bytes ({:.2f} KB)'.format(
        storage_centers, storage_centers / 1024))
    print('  Storage - metadata: {} bytes ({:.2f} KB)'.format(
        storage_meta, storage_meta / 1024))
    print('  Storage - index total: {} bytes ({:.2f} KB)'.format(
        storage_index_total, storage_index_total / 1024))
    print('  Storage - gallery estimate: {} bytes ({:.2f} MB)'.format(
        storage_gallery_estimate, storage_gallery_estimate / 1024 / 1024))
    print('  Storage inflation ratio: {:.4f}'.format(storage_inflation))
    print('  Peak RAM: {:.2f} MB'.format(peak_mem))

    # --- Save metrics JSON ---
    metrics = {
        'params': {
            'n_clusters': n_clusters,
            'key_size': args.key_size,
            'K': K,
            'n_init': 10,
            'random_state': 42,
        },
        'dataset': {
            'n_total_features': int(n_total),
            'n_gallery': int(n_gallery),
        },
        'offline_build_time': {
            'kmeans_time_s': round(cluster_duration, 4),
            'center_encrypt_time_s': round(enroll_duration, 4),
            'total_build_time_s': round(total_build_time, 4),
        },
        'cluster_size_stats': {
            'min': int(cluster_sizes.min()),
            'max': int(cluster_sizes.max()),
            'mean': round(float(cluster_sizes.mean()), 2),
            'std': round(float(cluster_sizes.std()), 2),
        },
        'storage_overhead': {
            'centers_bytes': int(storage_centers),
            'meta_bytes': int(storage_meta),
            'index_total_bytes': int(storage_index_total),
            'gallery_enrolled_estimate_bytes': int(storage_gallery_estimate),
            'inflation_ratio': round(storage_inflation, 6),
        },
        'peak_ram_mb': round(peak_mem, 2),
    }

    metrics_path = args.metrics_output
    if not metrics_path:
        metrics_path = os.path.join(args.folder, 'metrics_build.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print('[ClusterIndex] Metrics saved to {}'.format(metrics_path))


if __name__ == '__main__':
    main()
