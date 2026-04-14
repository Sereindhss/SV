#!/usr/bin/env python
import sys
import numpy as np
import damgard_jurik
import argparse
import os
import time
import random
import pickle
import hashlib
from itertools import repeat
import shutil
import resource
import json
import os.path
import gmpy2
from gmpy2 import mpz
from damgard_jurik import EncryptedNumber

# parse the args
parser = argparse.ArgumentParser(description='Enrollment in SecureVector (Damgard-Jurik Accelerated)')
parser.add_argument('--K', default=128, type=int)
parser.add_argument('--s', default=1, type=int, help='Damgard-Jurik s parameter')
parser.add_argument('--feat_list', type=str)
parser.add_argument('--folder', type=str,
                    help='use to store the keys and encrypted features')
parser.add_argument('--public_key', default='libs/SecureVector/keys/publickey',
                    type=str, help='path to the public key')
parser.add_argument('--key_size', default=2048, type=int)
parser.add_argument(
    '--r_pool_cache',
    action='store_true',
    help='Load/save offline R_i pool under --r_pool_cache_dir to skip recomputation on repeat runs',
)
parser.add_argument(
    '--r_pool_cache_dir',
    default='libs/SV_DJ/cache',
    type=str,
    help='Directory for R_pool .pkl files',
)
args = parser.parse_args()

def load_features(feature_list):
    features = []
    with open(feature_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        feature = [float(e) for e in parts[1:]]
        feature = feature/np.linalg.norm(np.array(feature))
        features.append(feature)
    return features

def offline_precompute_Ri(n_features, public_key):
    """
    【策略2：离线预计算】
    提前生成极耗时的大数随机幂 R_i = r^{N^s} mod N^{s+1}。
    这一步可以在服务器夜间空闲时（Offline）完成，不占用实时注册（Online）的时间。
    """
    print(f'[Offline Phase] 正在离线预计算 {n_features} 个大数随机幂 (策略2)...')
    start_offline = time.time()
    
    n_mpz = mpz(public_key.n)
    s = public_key.s
    n_s = n_mpz ** s
    n_s_plus_1 = n_mpz ** (s + 1)
    
    R_pool = []
    rand_gen = random.SystemRandom()
    for i in range(n_features):
        r = mpz(rand_gen.randrange(1, int(n_mpz)))
        # 【策略1：gmpy2 加速】底层 C 语言级快速模幂运算
        R_i = gmpy2.powmod(r, n_s, n_s_plus_1)
        R_pool.append(R_i)
        
    offline_time = time.time() - start_offline
    print(f'[Offline Phase] 离线预计算完成，耗时: {offline_time:.4f}s')
    return R_pool, offline_time


def _r_pool_cache_path(cache_dir, key_size, s, n_features, public_key):
    """Stable name per (key_size, s, n, n's modulus identity)."""
    n_bytes = str(int(public_key.n)).encode('utf-8')
    tag = hashlib.sha256(n_bytes).hexdigest()[:16]
    fname = 'r_pool_{}_{}_{}_{}.pkl'.format(key_size, s, n_features, tag)
    return os.path.join(cache_dir, fname)


def try_load_r_pool_cache(cache_path, n_features_expected):
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        if data.get('n') != n_features_expected:
            return None
        R_list = data['R_list']
        if len(R_list) != n_features_expected:
            return None
        return [mpz(x) for x in R_list]
    except Exception:
        return None


def save_r_pool_cache(cache_path, R_pool):
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    data = {
        'n': len(R_pool),
        'R_list': [int(x) for x in R_pool],
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[Offline Phase] R_pool 已写入缓存: {cache_path}')

def enroll(feature, K, L, M, public_key, precomputed_R_i):
    """
    在线加密特征 (利用预计算的 R_i)
    """
    start = time.time()
    u_list = [int(e) for e in np.random.rand(K)*(2*L)]
    v_list = [int(e) for e in np.random.rand(K)*(2*L)]
    s_list = [1 if v % 2 == 0 else -1 for v in v_list]

    # generate c_f
    n = len(feature)
    scale = [s_list[i] * np.e**((u_list[i]-L)/M) for i in range(K)]
    b_f = [x for item in scale for x in repeat(item, n//K)] * feature
    W_f = np.linalg.norm(b_f)
    c_f = b_f/W_f

    # encode
    base = [(4*L)**(K-1-i) for i in range(K)]
    w_f = int((np.log(W_f) + L/M)/(2*L/M) * 2**15 * L**8)
    C_f = np.dot(u_list, base) + \
        np.dot(v_list, base) * (4*L)**(K) + \
        w_f * (4*L)**(2*K)
    
    C_f = int(C_f)
    duration_plain = time.time() - start

    # --- 【策略1+2：在线极速加密】 ---
    start = time.time()
    
    n_mpz = mpz(public_key.n)
    s = public_key.s
    n_s_plus_1 = n_mpz ** (s + 1)
    g = n_mpz + 1  # Damgard-Jurik 的生成元 g = n + 1
    
    # 1. 计算极小明文的高次幂：(n+1)^m mod n^{s+1} (这步非常快)
    C_f_mpz = mpz(C_f)
    gm_m = gmpy2.powmod(g, C_f_mpz, n_s_plus_1)
    
    # 2. 乘上离线预计算好的巨大随机数幂 R_i (免去了在线大数幂运算)
    C_val = gmpy2.f_mod(gmpy2.mul(gm_m, precomputed_R_i), n_s_plus_1)
    
    # 3. 封装回原生库的 EncryptedNumber 对象，保证代码架构无缝衔接
    C_tilde_f = EncryptedNumber(int(C_val), public_key)
    
    duration_cypher = time.time() - start
    # ---------------------------------
    
    c_f_bits = C_f.bit_length()
    c_f_kb = c_f_bits / 1024  
    
    return [c_f, C_tilde_f], [duration_plain, duration_cypher, c_f_kb]
    

def main(K, L, M, s, key_size, feature_list, folder, public_key_path, use_r_pool_cache, r_pool_cache_dir):
    features = load_features(feature_list)
    n, dim = len(features), len(features[0])

    print('[SecureVector-DJ] Encrypting features...')
    publickey = np.load(public_key_path, allow_pickle=True)[0]
    if int(publickey.s) != int(s):
        raise SystemExit(
            'Public key s={} does not match --s {}. Remove stale keys under libs/SV_DJ/keys/ '
            'or regenerate with: python3 libs/SV_DJ/crypto_system.py --genkey 1 --key_size {} --s {}'.format(
                int(publickey.s), int(s), key_size, int(s)
            )
        )

    # 【新增】：进入真实的特征循环注册前，先提取预计算池（可选磁盘缓存）
    cache_path = _r_pool_cache_path(r_pool_cache_dir, key_size, s, n, publickey)
    R_pool = None
    offline_time = 0.0
    cache_hit = False
    if use_r_pool_cache:
        cached = try_load_r_pool_cache(cache_path, n)
        if cached is not None:
            R_pool = cached
            cache_hit = True
            print(f'[Offline Phase] 命中 R_pool 缓存，跳过预计算 ({cache_path})')
    if R_pool is None:
        R_pool, offline_time = offline_precompute_Ri(n, publickey)
        if use_r_pool_cache:
            try:
                save_r_pool_cache(cache_path, R_pool)
            except Exception as e:
                print(f'[Offline Phase] 警告: 写入 R_pool 缓存失败: {e}')

    start = time.time()
    duration_plain = []
    duration_cypher = []
    c_f_kb_list = []  
    
    for i, feature in enumerate(features):
        # 每次注册消耗一个预计算的随机数幂
        R_i = R_pool.pop()
        result, durations = enroll(feature, K, L, M, publickey, R_i)
        np.save('{}/{}.npy'.format(folder, i),
                np.array(result, np.dtype(object)))
        duration_plain.append(durations[0])
        duration_cypher.append(durations[1])
        c_f_kb_list.append(durations[2])
        if i % 1000 == 0:
            print('  Online Enrolled: {}/{}'.format(i, n))
            
    online_duration = time.time() - start
    
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    sample_file_size = 0
    if n > 0:
        sample_file_size = os.path.getsize('{}/0.npy'.format(folder)) / 1024.0
        
    avg_cf_kb = sum(c_f_kb_list) / n if n > 0 else 0

    print('\n[Online Phase] total duration {:.4f}s, permutation duration {:.4f}s, DJ fast duration {:.4f}s, encrypted {} features.'.format(
        online_duration, sum(duration_plain), sum(duration_cypher), n))
    print('[Metrics] Enrollment Peak RAM: {:.2f} MB'.format(peak_mem))
    print('[Metrics] Template File Size: {:.2f} KB'.format(sample_file_size))
    print('[Metrics] Average Compressed Integer Size: {:.2f} Kb\n'.format(avg_cf_kb))
    
    security_level = 2*K + K*np.log2(L)
    plaintext_bits = key_size * s
    
    metrics_path = os.path.join(folder, 'metrics_dj_enroll.json')
    metrics_data = {
        'security': {
            'security_bits': round(float(security_level), 2),
            'plaintext_space_bits': plaintext_bits,
            'key_size': key_size,
            's': s,
            'K': K,
            'L': int(L)
        },
        'efficiency': {
            'offline_precompute_time_ms': round(offline_time * 1000, 4), # 记录被分离出的离线耗时
            'avg_online_enrollment_time_ms': round((sum(duration_cypher) + sum(duration_plain)) / n * 1000, 4) if n > 0 else 0,
            'online_throughput_items_per_sec': round(n / online_duration, 2) if online_duration > 0 else 0,
            'r_pool_cache_hit': cache_hit,
            'r_pool_cache_path': cache_path if use_r_pool_cache else None,
        },
        'storage': {
            'template_file_size_kb': round(sample_file_size, 2),
            'avg_compressed_integer_size_kb': round(avg_cf_kb, 2)
        },
        'peak_ram_mb': round(peak_mem, 2)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f'[Metrics] Saved enrollment metrics to {metrics_path}')

if __name__ == '__main__':
    L = int(np.ceil(2**((args.key_size * args.s)/(2*args.K+9)-2) - 1))
    M = L/128
    security_level = 2*args.K + args.K*np.log2(L)

    print('K: {}   L: {}   M: {}   s: {}'.format(args.K, L, M, args.s))
    print('the security level is: {:.2f} bits'.format(security_level))
    assert L > 1
    if os.path.exists(args.folder):
        shutil.rmtree(args.folder)
    os.makedirs(args.folder)

    main(
        args.K,
        L,
        M,
        args.s,
        args.key_size,
        args.feat_list,
        args.folder,
        '{}_{}_s{}.npy'.format(args.public_key, args.key_size, args.s),
        args.r_pool_cache,
        args.r_pool_cache_dir,
    )
