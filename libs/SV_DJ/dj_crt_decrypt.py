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
import gmpy2
from damgard_jurik.utils import crm, inv_mod
from gmpy2 import mpz

# Reuse p^(s+1), q^(s+1), n^(s+1) across many pow_mod_crt calls (same decrypt session).
_CRT_MODULI_CACHE: dict[tuple[int, int, int, int], tuple[mpz, mpz, mpz]] = {}

# Lagrange coefficients per PrivateKeyRing instance (stable id in long-running match).
_LAM_COEFF_CACHE: dict[int, tuple[int, ...]] = {}


def _crt_moduli(n: int, p: int, q: int, s: int) -> tuple[mpz, mpz, mpz]:
    key = (int(n), int(p), int(q), int(s))
    hit = _CRT_MODULI_CACHE.get(key)
    if hit is not None:
        return hit
    pp = mpz(p) ** (s + 1)
    qq = mpz(q) ** (s + 1)
    m_mod = mpz(n) ** (s + 1)
    if m_mod != pp * qq:
        raise ValueError(
            "Expected n^(s+1)==p^(s+1)q^(s+1), got {} vs {}".format(int(m_mod), int(pp * qq))
        )
    _CRT_MODULI_CACHE[key] = (pp, qq, m_mod)
    return pp, qq, m_mod


def _crt_two_residues_gmpy(y_p: mpz, y_q: mpz, pp: mpz, qq: mpz, m_mod: mpz) -> mpz:
    """
    CRT for coprime moduli pp, qq with gcd(pp,qq)=1: unique x mod pp*qq = m_mod
    with x ≡ y_p (mod pp) and x ≡ y_q (mod qq). Garner-style (one gmpy2.invert).
    """
    t = gmpy2.sub(y_q, y_p)
    inv_pp = gmpy2.invert(gmpy2.f_mod(pp, qq), qq)
    h = gmpy2.f_mod(gmpy2.mul(t, inv_pp), qq)
    return gmpy2.f_mod(gmpy2.add(y_p, gmpy2.mul(pp, h)), m_mod)


def pow_mod_crt(c: int, e: int, n: int, p: int, q: int, s: int) -> int:
    """
    Return pow(c, e, n^(s+1)) using CRT on p^(s+1) and q^(s+1).

    Correctness: for M = n^(s+1) = p^(s+1) q^(s+1), gcd(p^{s+1}, q^{s+1}) = 1,
    and pow(c, e, M) mod p^{s+1} = pow(c mod p^{s+1}, e, p^{s+1}).
    """
    pp, qq, m_mod = _crt_moduli(int(n), int(p), int(q), int(s))
    c_z = mpz(int(c))
    e_z = mpz(int(e))
    y_p = gmpy2.powmod(gmpy2.f_mod(c_z, pp), e_z, pp)
    y_q = gmpy2.powmod(gmpy2.f_mod(c_z, qq), e_z, qq)
    r = _crt_two_residues_gmpy(y_p, y_q, pp, qq, m_mod)
    return int(r)


def lagrange_coefficients_for_ring(ring: PrivateKeyRing) -> tuple[int, ...]:
    """
    Precompute lam(i) for each party index i in ring.i_list (same as PrivateKeyRing.decrypt).
    Cached per ring object id to avoid O(threshold^2) work on every decrypt.
    """
    rid = id(ring)
    hit = _LAM_COEFF_CACHE.get(rid)
    if hit is not None:
        return hit
    pk = ring.public_key
    n_s_m = int(pk.n_s_m)
    l0 = int(pk.delta) % n_s_m
    coeffs: list[int] = []
    for i in ring.i_list:
        s_set = ring.S - {i}
        l = l0
        for i_prime in s_set:
            l = (l * i_prime * inv_mod(i_prime - i, n_s_m)) % n_s_m
        coeffs.append(int(l))
    tup = tuple(coeffs)
    _LAM_COEFF_CACHE[rid] = tup
    return tup


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

    c_val = int(c.value) if hasattr(c, "value") else int(c)

    c_list = []
    for share in ring.private_key_shares:
        e = int(share.two_delta_s_i)
        c_list.append(pow_mod_crt(c_val, e, n, p, q, s))

    lam_coeffs = lagrange_coefficients_for_ring(ring)

    c_prime = mpz(1)
    n_s_1_mpz = mpz(n_s_1)
    for c_i, lam_i in zip(c_list, lam_coeffs):
        exp = 2 * lam_i
        t = mpz(pow_mod_crt(int(c_i), exp, n, p, q, s))
        c_prime = gmpy2.f_mod(gmpy2.mul(c_prime, t), n_s_1_mpz)

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
    Same as damgard_jurik.keygen, plus writes factors_{key_size_label}_s{s}.json
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
    factors_path = factors_json_path(keys_dir, key_size_label, s)
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


def factors_json_path(keys_dir: str, key_size_label: int, s: int) -> str:
    return os.path.join(keys_dir, f"factors_{key_size_label}_s{s}.json")


def load_factors_json(keys_dir: str, key_size_label: int, s: int = 1) -> Optional[Tuple[int, int, int]]:
    path = factors_json_path(keys_dir, key_size_label, s)
    if not os.path.isfile(path):
        # Fallback to legacy name
        path = os.path.join(keys_dir, f"factors_{key_size_label}.json")
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
