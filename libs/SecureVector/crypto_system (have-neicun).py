#!/usr/bin/env python
"""
This is the "root" code, only it can get access to the private key.
    In: paths to two enrolled features
    Out: Similarity scores
useage:
    generate keys:
        python crypo_system.py --genkey 1 --key_size 1024 
    calculate similarities:
        python crypo_system.py --key_size 1024 --K 128 --folder $F --pair_list $P --score_list $S
"""
import sys
import math
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import random
import resource
from itertools import repeat
from joblib import Parallel, delayed
import tracemalloc

parser = argparse.ArgumentParser(description='Match in SecureVector')
parser.add_argument('--folder', default='', type=str,
                    help='fold which stores the encrypted features')
parser.add_argument('--pair_list', default='', type=str, help='pair file')
parser.add_argument('--score_list', type=str,
                    help='a file which stores the scores')
parser.add_argument('--K', default=128, type=int)
parser.add_argument('--key_size', default=1024, type=int)
parser.add_argument('--genkey', default=0, type=int)
args = parser.parse_args()

if args.genkey == 1:
    pubkey, prikey = paillier.generate_paillier_keypair(n_length=args.key_size)
    np.save('libs/SecureVector/keys/privatekey_{}.npy'.format(args.key_size), [prikey])
    np.save('libs/SecureVector/keys/publickey_{}.npy'.format(args.key_size), [pubkey])
    exit(1)
else:
    private_key = np.load('libs/SecureVector/keys/privatekey_{}.npy'.format(args.key_size), allow_pickle=True)[0]


def load_enrolled_file(file):
    c_f, C_tilde_f = np.load(file, allow_pickle=True)
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
    # 解密（不计入内存追踪）
    start = time.time()
    C_z = decrypt_sum(C_tilde_x, C_tilde_y)
    duration_cypher = time.time() - start

    # 开始追踪内存：仅相似度计算部分（避免大型数组分配）
    tracemalloc.start()
    start = time.time()

    n = len(c_x)
    seg_len = n // K
    # 直接计算分段点积，不生成 c_x*c_y 数组
    bar_c_xy = []
    for i in range(0, n, seg_len):
        bar_c_xy.append(np.dot(c_x[i:i+seg_len], c_y[i:i+seg_len]))

    u_list, v_list, w_z = decode_uvw(C_z, K, L)
    s_list = [1 if v % 2 == 0 else -1 for v in v_list]

    W_z = np.e**((w_z - 2**15 * L**8) / (2**14 * L**7 * M))
    score = W_z * sum(
        bar_c_xy[i] / (s_list[i] * np.e**((u_list[i] - 2 * L) / M))
        for i in range(K)
    )

    duration_plain = time.time() - start
    current, peak = tracemalloc.get_traced_memory()  # 峰值内存（字节）
    tracemalloc.stop()

    return score, [duration_plain, duration_cypher, peak]


def main(folder, pair_list, score_list, K, L, M):
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    fw = open(score_list, 'w')
    print('[SecureVector] Decrypting features...')
    start = time.time()
    duration_plain = []
    duration_cypher = []
    match_mems = []

    n = len(lines)
    for i, line in enumerate(lines):
        file1, file2, _ = line.strip().split(' ')
        c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
        c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
        score, durations = calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
        duration_plain.append(durations[0])
        duration_cypher.append(durations[1])
        match_mems.append(durations[2])
        fw.write('{} {} {}\n'.format(file1, file2, score))
        if i % 100000 == 0:
            print('{}/{}'.format(i, n))

    fw.close()
    duration = time.time() - start

    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    avg_match_bytes = sum(match_mems) / n
    avg_match_kb = avg_match_bytes * 8 / 1024

    print('total duration {}, permutation duration {}, paillier duration {}, calculate {} pairs.\n'.format(
        duration, sum(duration_plain), sum(duration_cypher), n))
    print('[Metrics] Matching Peak RAM (匹配峰值内存): {:.2f} MB'.format(peak_mem))
    print('[Metrics] Average Match Memory (平均匹配内存): {:.2f} Kb'.format(avg_match_kb))


if __name__ == '__main__':
    L = int(np.ceil(2 ** (args.key_size / (2 * args.K + 9) - 2) - 1))
    M = L / 128
    security_level = 2 * args.K + args.K * np.log2(L)

    assert L > 1
    main(args.folder, args.pair_list, args.score_list, args.K, L, M)
