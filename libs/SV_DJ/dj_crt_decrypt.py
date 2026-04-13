#!/usr/bin/env python3
"""
CRT-accelerated modular exponentiation for Damgård–Jurik threshold decryption.

For modulus M = n^(s+1) = p^(s+1) q^(s+1) with n = pq, any pow(c, e, M) can be
computed via two half-width modexps and CRT recombination — same result as
pow(c, e, M), without changing the cryptographic scheme.

Uses the same threshold PrivateKeyRing.decrypt flow as damgard_jurik, but
replaces large moduli pow() with pow_mod_crt for the dominant operations.
"""
from __future__ import annotations

import json
import os
from math import factorial
from typing import Any, Optional, Tuple

from damgard_jurik.crypto import (
    PrivateKeyRing,
    PrivateKeyShare,
    PublicKey,
    damgard_jurik_reduce,
)
from damgard_jurik.prime_gen import gen_safe_prime_pair
from damgard_jurik.shamir import share_secret
from damgard_jurik.utils import crm, int_to_mpz, inv_mod
from gmpy2 import mpz


def pow_mod_crt(c: int, e: int, n: int, p: int, q: int, s: int) -> int:
    """
    Return pow(c, e, n^(s+1)) using CRT on p^(s+1) and q^(s+1).

    Correctness: for M = n^(s+1) = p^(s+1) q^(s+1), gcd(p^{s+1}, q^{s+1}) = 1,
    and pow(c, e, M) mod p^{s+1} = pow(c mod p^{s+1}, e, p^{s+1}).
    """
    c, e, n, p, q = int(c), int(e), int(n), int(p), int(q)
    pp = pow(p, s + 1)
    qq = pow(q, s + 1)
    m_mod = pow(n, s + 1)
    if m_mod != pp * qq:
        raise ValueError(f"Expected n^(s+1)==p^(s+1)q^(s+1), got {m_mod} vs {pp * qq}")
    y_p = pow(c % pp, e, pp)
    y_q = pow(c % qq, e, qq)
    return int(crm([y_p, y_q], [pp, qq]) % m_mod)


def decrypt_encrypted_number_crt(c: Any, ring: PrivateKeyRing, p: int, q: int) -> int:
    """
    Threshold DJ decrypt: mathematically same as PrivateKeyRing.decrypt(c),
    but pow_mod on n^(s+1) replaced by pow_mod_crt where applicable.
    """
    pk = ring.public_key
    n = int(pk.n)
    s = int(pk.s)
    n_s_1 = int(pk.n_s_1)
    n_s = int(pk.n_s)
    n_s_m = int(pk.n_s_m)

    c_val = int(c.value) if hasattr(c, "value") else int(c)

    c_list = []
    for share in ring.private_key_shares:
        e = int(share.two_delta_s_i)
        c_list.append(pow_mod_crt(c_val, e, n, p, q, s))

    @int_to_mpz
    def lam(i: int) -> int:
        s_set = ring.S - {i}
        l = pk.delta % n_s_m
        for i_prime in s_set:
            l = l * i_prime * inv_mod(i_prime - i, n_s_m) % n_s_m
        return l

    c_prime = mpz(1)
    for c_i, i in zip(c_list, ring.i_list):
        exp = 2 * int(lam(i))
        c_prime = (c_prime * pow_mod_crt(int(c_i), exp, n, p, q, s)) % n_s_1

    c_prime = damgard_jurik_reduce(c_prime, pk.s, pk.n)
    m = c_prime * ring.inv_four_delta_squared % n_s
    return int(m)


def keygen_save_factors(
    n_bits: int,
    s: int,
    key_size_label: int,
    keys_dir: str,
    threshold: int = 3,
    n_shares: int = 3,
) -> Tuple[PublicKey, PrivateKeyRing]:
    """
    Same as damgard_jurik.keygen, plus writes factors_{key_size_label}.json
    with p, q (decimal strings) for CRT helpers.
    """
    if n_bits < 16:
        raise ValueError("Minimum number of bits for encryption is 16")
    if s < 1:
        raise ValueError("s must be >= 1")
    if n_shares < threshold or threshold < 1:
        raise ValueError("Invalid threshold / n_shares")

    p, q = gen_safe_prime_pair(n_bits)
    p_prime, q_prime = (p - 1) // 2, (q - 1) // 2
    n, m = p * q, p_prime * q_prime
    n_s = n ** s
    n_s_m = n_s * m
    d_secret = crm(a_list=[0, 1], n_list=[m, n_s])
    shares = share_secret(
        secret=d_secret, modulus=n_s_m, threshold=threshold, n_shares=n_shares
    )
    delta = factorial(n_shares)
    public_key = PublicKey(n=n, s=s, m=m, threshold=threshold, delta=delta)
    private_key_shares = [
        PrivateKeyShare(public_key=public_key, i=i, s_i=s_i) for i, s_i in shares
    ]
    private_key_ring = PrivateKeyRing(private_key_shares)

    os.makedirs(keys_dir, exist_ok=True)
    factors_path = os.path.join(keys_dir, "factors_{}.json".format(key_size_label))
    with open(factors_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "p": str(int(p)),
                "q": str(int(q)),
                "s": s,
                "n_bits": n_bits,
                "threshold": threshold,
                "n_shares": n_shares,
            },
            f,
            indent=2,
        )

    return public_key, private_key_ring


def load_factors_json(keys_dir: str, key_size_label: int) -> Optional[Tuple[int, int, int]]:
    path = os.path.join(keys_dir, "factors_{}.json".format(key_size_label))
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return int(d["p"]), int(d["q"]), int(d["s"])


def verify_crt_matches_reference(
    ring: PrivateKeyRing, p: int, q: int, pub: Any
) -> None:
    """Encrypt a small test message and compare decrypt vs decrypt_encrypted_number_crt."""
    m_test = 42
    c = pub.encrypt(m_test)
    ref = int(ring.decrypt(c))
    fast = int(decrypt_encrypted_number_crt(c, ring, p, q))
    if ref != fast:
        raise RuntimeError(
            "CRT decrypt mismatch: ref={} fast={} (do not use --crt_decrypt)".format(
                ref, fast
            )
        )
