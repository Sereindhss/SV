#!/usr/bin/env python
import os
import sys

# Support: `python3 libs/SV_DJ/crypto_system.py` from repo root (so `libs` is importable)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
import multiprocessing
import numpy as np
import argparse
import time
import json
import resource
import cProfile
import pstats
import io
from itertools import repeat
import gmpy2
from gmpy2 import mpz
from damgard_jurik import EncryptedNumber

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Match in SecureVector (Damgard-Jurik Accelerated)')
    parser.add_argument('--folder', default='', type=str,
                        help='fold which stores the encrypted features')
    parser.add_argument('--pair_list', default='', type=str, help='pair file')
    parser.add_argument('--feat_list', default='', type=str, help='original plaintext feature file for error checking')
    parser.add_argument('--score_list', type=str, default='score.list',
                        help='a file which stores the scores')
    parser.add_argument('--K', default=128, type=int)
    parser.add_argument('--key_size', default=1024, type=int)
    parser.add_argument('--s', default=1, type=int, help='Damgard-Jurik s parameter')
    parser.add_argument('--genkey', default=0, type=int)
    parser.add_argument(
        '--crt_decrypt',
        action='store_true',
        help='Use CRT-accelerated threshold decrypt (requires libs/SV_DJ/keys/factors_<key_size>.json from genkey)',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Parallel worker processes for matching (default 1). Uses pool initializer (works with spawn).',
    )
    parser.add_argument(
        '--profile_pairs',
        type=int,
        default=0,
        metavar='N',
        help='If N>0, run cProfile on the first N pairs only, print hot functions, then exit (no full score file).',
    )
    return parser


_CRT_UNSET = object()

KEYS_DIR = 'libs/SV_DJ/keys'

private_key = None
_crt_factors = None


def load_private_key(key_size, keys_dir=KEYS_DIR):
    global private_key
    private_key = np.load(
        os.path.join(keys_dir, 'privatekey_{}.npy'.format(key_size)), allow_pickle=True)[0]
    return private_key


def _pool_init_worker(keys_dir, key_size, repo_root, crt_tuple):
    """Pool initializer: load keys in each worker (spawn-safe)."""
    global private_key, _crt_factors
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    private_key = np.load(
        os.path.join(keys_dir, 'privatekey_{}.npy'.format(key_size)), allow_pickle=True)[0]
    _crt_factors = crt_tuple

def load_enrolled_file(file):
    c_f, C_tilde_f = np.load(file, allow_pickle=True)
    return c_f, C_tilde_f

def load_original_features(feat_list_path):
    features = {}
    if not feat_list_path or not os.path.exists(feat_list_path):
        return features
    with open(feat_list_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(' ')
            feature = np.array([float(e) for e in parts[1:]])
            feature = feature / np.linalg.norm(feature)
            features[str(i)] = feature
    return features

def decode_uvw(C_f, K, L):
    u_list, v_list = [], []
    for i in range(K):
        next_C_f = C_f//(4*L)
        u_list.append(C_f - (4*L)*next_C_f)
        C_f = next_C_f
    for i in range(K):
        next_C_f = C_f//(4*L)
        v_list.append(C_f - (4*L)*next_C_f)
        C_f = next_C_f
    w_f = C_f
    u_list.reverse()
    v_list.reverse()
    return u_list, v_list, int(w_f)

# 将 crypto_system.py 里的 calculate_sim 函数替换为如下代码：

def calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M, priv=None, crt_tuple=_CRT_UNSET):
    """
    crt_tuple: use _CRT_UNSET to follow global _crt_factors; pass None for no CRT when priv is set (workers).
    """
    pk = private_key if priv is None else priv
    if crt_tuple is _CRT_UNSET:
        ct = _crt_factors
    else:
        ct = crt_tuple

    # homomorphic add + decrypt
    start = time.time()
    t_add_start = time.time()
    
    # 【修复点1：恢复安全、原生的同态加法】
    # 加法底层仅为一次模乘，本身耗时极短（<1ms），无需强行破解内部数据结构
    C_sum = C_tilde_x + C_tilde_y
    t_add = time.time() - t_add_start
    
    # 解密过程（可选 CRT 加速，与 PrivateKeyRing.decrypt 数学等价）
    if ct is not None:
        from libs.SV_DJ.dj_crt_decrypt import decrypt_encrypted_number_crt

        C_z = decrypt_encrypted_number_crt(C_sum, pk, ct[0], ct[1])
    else:
        C_z = pk.decrypt(C_sum)
    duration_cypher = time.time() - start

    # generate bar_c_xy
    start = time.time()
    c_xy = c_x * c_y
    n = len(c_x)
    bar_c_xy = [sum(c_xy[i:i+n//K]) for i in range(0, n, n//K)]

    # 强制将 C_z 转为 int，避免解码器类型错误
    u_list, v_list, w_z = decode_uvw(int(C_z), K, L)
    s_list = [1 if v % 2 == 0 else -1 for v in v_list]
    
    # calculate the score
    W_z = np.e**((w_z - 2**15 * L**8)/(2**14 * L**7*M))
    score = W_z * sum([bar_c_xy[i]/(s_list[i] * np.e**((u_list[i]-2*L)/M)) for i in range(K)])
    duration_plain = time.time() - start
    
    # 【修复点2：极其安全的密文大小获取方式】
    # 用 try-except 拦截所有潜在的库属性不兼容问题，确保不影响核心流程
    C_sum_size = 0
    try:
        if hasattr(C_sum, 'value'):
            C_sum_size = int(C_sum.value).bit_length() / 8
        elif hasattr(C_sum, 'c'):
            C_sum_size = int(C_sum.c).bit_length() / 8
    except Exception:
        pass # 如果提取失败，默认记录 0，绝对不让程序崩溃

    return score, [duration_plain, duration_cypher, t_add, C_sum_size]

def main(folder, pair_list, score_list, K, L, M, s, key_size, feat_list):
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    fw = open(score_list, 'w')
    print('[SecureVector-DJ] Decrypting and matching features...')
    
    orig_features = load_original_features(feat_list)

    start = time.time()
    duration_plain = []
    duration_cypher = []
    duration_homo_add = []
    c_sum_sizes = []
    errors = []

    n = len(lines)
    
    for i, line in enumerate(lines):
        file1, file2, _ = line.strip().split(' ')
        c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
        c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
        
        score, durations = calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
        
        duration_plain.append(durations[0])
        duration_cypher.append(durations[1])
        duration_homo_add.append(durations[2])
        c_sum_sizes.append(durations[3])
        
        fw.write('{} {} {}\n'.format(file1, file2, score))
        
        if orig_features:
            x_orig = orig_features.get(file1)
            y_orig = orig_features.get(file2)
            if x_orig is not None and y_orig is not None:
                orig_score = np.dot(x_orig, y_orig)
                errors.append(abs(score - orig_score))
                
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))

    fw.close()
    duration = time.time() - start
    
    print('total duration {:.4f}s, permutation duration {:.4f}s, DJ duration {:.4f}s, calculate {} pairs.\n'.format(
        duration, sum(duration_plain), sum(duration_cypher), n))
        
    avg_homo_add_ms = sum(duration_homo_add) / n * 1000 if n > 0 else 0
    avg_cypher_ms = sum(duration_cypher) / n * 1000 if n > 0 else 0
    avg_plain_ms = sum(duration_plain) / n * 1000 if n > 0 else 0
    avg_total_match_ms = avg_cypher_ms + avg_plain_ms
    avg_c_sum_kb = sum(c_sum_sizes) / n / 1024 if n > 0 else 0
    avg_error = np.mean(errors) if errors else 0.0

    print(f'[Metrics] Avg fast homomorphic add time: {avg_homo_add_ms:.4f} ms')
    print(f'[Metrics] Avg match time (decrypt + plain): {avg_total_match_ms:.4f} ms')
    print(f'[Metrics] Avg similarity ciphertext size: {avg_c_sum_kb:.2f} KB')
    if errors:
        print(f'[Metrics] Avg recovery absolute error: {avg_error:.4e}')
        
    enroll_metrics_path = os.path.join(folder, 'metrics_dj_enroll.json')
    combined_metrics = {}
    if os.path.exists(enroll_metrics_path):
        with open(enroll_metrics_path, 'r') as f:
            combined_metrics = json.load(f)
    else:
        combined_metrics = {'security': {}, 'efficiency': {}, 'storage': {}}
        
    if 'efficiency' not in combined_metrics:
        combined_metrics['efficiency'] = {}
    combined_metrics['efficiency']['avg_match_time_ms'] = round(avg_total_match_ms, 4)
    combined_metrics['efficiency']['avg_homo_add_time_ms'] = round(avg_homo_add_ms, 4)
    combined_metrics['efficiency']['match_throughput_pairs_per_sec'] = round(n / duration, 2) if duration > 0 else 0
    
    if 'storage' not in combined_metrics:
        combined_metrics['storage'] = {}
    combined_metrics['storage']['similarity_ciphertext_size_kb'] = round(avg_c_sum_kb, 2)
    
    pub_key_path = 'libs/SV_DJ/keys/publickey_{}.npy'.format(key_size)
    priv_key_path = 'libs/SV_DJ/keys/privatekey_{}.npy'.format(key_size)
    if os.path.exists(pub_key_path) and os.path.exists(priv_key_path):
        combined_metrics['storage']['public_key_kb'] = round(os.path.getsize(pub_key_path) / 1024, 2)
        combined_metrics['storage']['private_key_kb'] = round(os.path.getsize(priv_key_path) / 1024, 2)
        
    if 'accuracy' not in combined_metrics:
        combined_metrics['accuracy'] = {}
    if errors:
        combined_metrics['accuracy']['avg_recovery_absolute_error'] = float(avg_error)
        
    final_metrics_path = os.path.join(folder, 'metrics_dj.json')
    with open(final_metrics_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4)
        
    print(f'[Metrics] Saved final combined metrics to {final_metrics_path}')


def _worker_match_one(payload):
    """One pair; uses globals private_key / _crt_factors set by pool initializer."""
    line, folder, K, L, M = payload
    file1, file2, _ = line.strip().split(' ')
    c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
    c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
    score, durations = calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
    return file1, file2, score, durations


def main_parallel(folder, pair_list, score_list, K, L, M, s, key_size, feat_list, jobs):
    """Multiprocessing match (same outputs as main; recovery error metrics omitted when jobs>1)."""
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    print('[SecureVector-DJ] Decrypting and matching features ({} workers)...'.format(jobs))
    work = [(ln, folder, K, L, M) for ln in lines]
    initargs = (KEYS_DIR, key_size, _REPO_ROOT, _crt_factors)

    start = time.time()
    chunksize = max(1, len(work) // (jobs * 8))
    with multiprocessing.Pool(jobs, initializer=_pool_init_worker, initargs=initargs) as pool:
        raw = pool.map(_worker_match_one, work, chunksize=chunksize)

    duration = time.time() - start
    duration_plain = [r[3][0] for r in raw]
    duration_cypher = [r[3][1] for r in raw]
    duration_homo_add = [r[3][2] for r in raw]
    c_sum_sizes = [r[3][3] for r in raw]
    n = len(raw)

    with open(score_list, 'w') as fw:
        for file1, file2, score, _durs in raw:
            fw.write('{} {} {}\n'.format(file1, file2, score))

    print('total duration {:.4f}s, permutation duration {:.4f}s, DJ duration {:.4f}s, calculate {} pairs.\n'.format(
        duration, sum(duration_plain), sum(duration_cypher), n))
    avg_homo_add_ms = sum(duration_homo_add) / n * 1000 if n > 0 else 0
    avg_cypher_ms = sum(duration_cypher) / n * 1000 if n > 0 else 0
    avg_plain_ms = sum(duration_plain) / n * 1000 if n > 0 else 0
    avg_total_match_ms = avg_cypher_ms + avg_plain_ms
    avg_c_sum_kb = sum(c_sum_sizes) / n / 1024 if n > 0 else 0

    print(f'[Metrics] Avg fast homomorphic add time: {avg_homo_add_ms:.4f} ms')
    print(f'[Metrics] Avg match time (decrypt + plain): {avg_total_match_ms:.4f} ms')
    print(f'[Metrics] Avg similarity ciphertext size: {avg_c_sum_kb:.2f} KB')
    print('[Metrics] (parallel mode) avg_recovery_absolute_error not computed')

    enroll_metrics_path = os.path.join(folder, 'metrics_dj_enroll.json')
    combined_metrics = {}
    if os.path.exists(enroll_metrics_path):
        with open(enroll_metrics_path, 'r') as f:
            combined_metrics = json.load(f)
    else:
        combined_metrics = {'security': {}, 'efficiency': {}, 'storage': {}}
    if 'efficiency' not in combined_metrics:
        combined_metrics['efficiency'] = {}
    combined_metrics['efficiency']['avg_match_time_ms'] = round(avg_total_match_ms, 4)
    combined_metrics['efficiency']['avg_homo_add_time_ms'] = round(avg_homo_add_ms, 4)
    combined_metrics['efficiency']['match_throughput_pairs_per_sec'] = round(n / duration, 2) if duration > 0 else 0
    combined_metrics['efficiency']['match_parallel_jobs'] = jobs
    if 'storage' not in combined_metrics:
        combined_metrics['storage'] = {}
    combined_metrics['storage']['similarity_ciphertext_size_kb'] = round(avg_c_sum_kb, 2)
    pub_key_path = 'libs/SV_DJ/keys/publickey_{}.npy'.format(key_size)
    priv_key_path = 'libs/SV_DJ/keys/privatekey_{}.npy'.format(key_size)
    if os.path.exists(pub_key_path) and os.path.exists(priv_key_path):
        combined_metrics['storage']['public_key_kb'] = round(os.path.getsize(pub_key_path) / 1024, 2)
        combined_metrics['storage']['private_key_kb'] = round(os.path.getsize(priv_key_path) / 1024, 2)
    final_metrics_path = os.path.join(folder, 'metrics_dj.json')
    with open(final_metrics_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4)
    print(f'[Metrics] Saved final combined metrics to {final_metrics_path}')


def run_profile_subset(folder, pair_list, K, L, M, n_pairs):
    """cProfile first n_pairs pairs; print cumulative time stats."""
    with open(pair_list, 'r') as f:
        lines = f.readlines()[:n_pairs]
    print('[SV_DJ] Profiling first {} pairs (cProfile)...'.format(len(lines)))

    def _run():
        for i, line in enumerate(lines):
            file1, file2, _ = line.strip().split(' ')
            c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
            c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
            calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
            if i % 50 == 0 and i:
                print('  profiled {}/{}'.format(i, len(lines)))

    pr = cProfile.Profile()
    pr.enable()
    _run()
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
    ps.print_stats(45)
    print(buf.getvalue())
    print('[SV_DJ] Profile done. Look for damgard_jurik_reduce, pow, decrypt_encrypted_number_crt, etc.')


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    if args.genkey == 1:
        from libs.SV_DJ.dj_crt_decrypt import keygen_save_factors

        pubkey, prikey = keygen_save_factors(
            n_bits=args.key_size,
            s=args.s,
            key_size_label=args.key_size,
            keys_dir=KEYS_DIR,
        )

        os.makedirs(KEYS_DIR, exist_ok=True)
        np.save(os.path.join(KEYS_DIR, 'privatekey_{}.npy'.format(args.key_size)), [prikey])
        np.save(os.path.join(KEYS_DIR, 'publickey_{}.npy'.format(args.key_size)), [pubkey])

        pub_size = os.path.getsize(os.path.join(KEYS_DIR, 'publickey_{}.npy'.format(args.key_size))) / 1024
        priv_size = os.path.getsize(os.path.join(KEYS_DIR, 'privatekey_{}.npy'.format(args.key_size))) / 1024
        print(f"Keys generated. Public key: {pub_size:.2f} KB, Private key: {priv_size:.2f} KB")
        print(f"Factors for CRT decrypt saved to {KEYS_DIR}/factors_{args.key_size}.json")
        sys.exit(0)

    L = int(np.ceil(2**((args.key_size * args.s)/(2*args.K+9)-2) - 1))
    M = L/128
    assert L > 1

    load_private_key(args.key_size)

    if args.crt_decrypt:
        from libs.SV_DJ.dj_crt_decrypt import load_factors_json, verify_crt_matches_reference

        fac = load_factors_json(KEYS_DIR, args.key_size)
        if fac is None:
            raise SystemExit(
                'CRT decrypt requires {} — regenerate keys with: python3 libs/SV_DJ/crypto_system.py --genkey 1 --key_size {} --s {}'.format(
                    os.path.join(KEYS_DIR, 'factors_{}.json'.format(args.key_size)),
                    args.key_size,
                    args.s,
                )
            )
        p, q, s_fac = fac
        if s_fac != args.s:
            raise SystemExit('Mismatch: factors.json has s={} but args.s={}'.format(s_fac, args.s))
        pubkey = np.load(
            os.path.join(KEYS_DIR, 'publickey_{}.npy'.format(args.key_size)), allow_pickle=True
        )[0]
        verify_crt_matches_reference(private_key, p, q, pubkey)
        _crt_factors = (p, q)
        print('[SV_DJ] CRT-accelerated decrypt enabled (self-check passed).')

    if args.profile_pairs > 0:
        run_profile_subset(args.folder, args.pair_list, args.K, L, M, args.profile_pairs)
        sys.exit(0)

    if args.jobs > 1:
        main_parallel(
            args.folder, args.pair_list, args.score_list, args.K, L, M, args.s, args.key_size,
            args.feat_list, args.jobs,
        )
    else:
        main(args.folder, args.pair_list, args.score_list, args.K, L, M, args.s, args.key_size, args.feat_list)
