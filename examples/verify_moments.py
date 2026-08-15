"""
Verify the moment identities in src/moments.py -- sigma(i,j) (covariance),
mu3(i,j,k) (third joint cumulant), and kappa4(i,j,k,l) (fourth joint
cumulant) of the moment-vector components (X, X^2, W, WX, Y, XY, WY,
X^2 W, XWY) against direct Monte Carlo estimates, across several
different (X, W, Y) DGPs.

These identities are the only things mse_expansion.py and
coverage_cumulants.py assume about the underlying joint distribution:
W in {0,1} and E[X] = 0. 

Run with:

    python examples/verify_moments.py
"""
import os
import sys
from itertools import combinations_with_replacement

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.moments import sigma, mu3, kappa4, NONZERO_GRADIENT_INDICES, _CANONICAL_RAW_MOMENTS

N = 2_000_000
N_BATCHES = 25
Z_THRESHOLD = 2.0  

def dgp_normal_homoskedastic(rng, n):
    X = rng.normal(0, 1, n)
    W = rng.binomial(1, 0.5, n)
    Y = 0.5 + 1.0 * W + 0.9 * X + 0.3 * W * X + rng.normal(0, 1, n)
    return X, W, Y


def dgp_skewed_heteroskedastic(rng, n):
    X = rng.gamma(2.0, 1.0, n) - 2.0
    W = rng.binomial(1, 0.5, n)
    Y = -0.2 + 1.3 * W + 0.6 * X + 0.5 * W * X + rng.normal(0, 1, n) * (1 + np.abs(X))
    return X, W, Y


def dgp_heavy_tailed_arm_asymmetric(rng, n):
    X = rng.laplace(0, 0.6, n)
    W = rng.binomial(1, 0.5, n)
    scale = (0.6 + 0.8 * W) * (1 + 0.4 * np.abs(X))
    Y = 0.1 + 0.8 * W - 0.4 * X + 0.7 * W * X + rng.normal(0, 1, n) * scale
    return X, W, Y


def dgp_lognormal_misspecified(rng, n):
    X = rng.lognormal(0, 0.7, n) - np.exp(0.7 ** 2 / 2)
    W = rng.binomial(1, 0.5, n)
    Y = 1.0 * W + 0.5 * X + 0.2 * W * X + 0.1 * (X ** 2 - np.var(X)) + rng.normal(0, 1, n)
    return X, W, Y


DGPS = {
    "Normal, homoskedastic": dgp_normal_homoskedastic,
    "Skewed (Gamma), heteroskedastic": dgp_skewed_heteroskedastic,
    "Heavy-tailed (Laplace), arm-asymmetric variance": dgp_heavy_tailed_arm_asymmetric,
    "LogNormal, quadratic misspecification": dgp_lognormal_misspecified,
}

print("Precomputing symbolic sigma(i,j) for all 45 index pairs...")
pairs = list(combinations_with_replacement(range(1, 10), 2))
sigma_formulas = {(i, j): sigma(i, j) for (i, j) in pairs}

print("Precomputing symbolic mu3(i,j,k) for all 165 index triples...")
triples = list(combinations_with_replacement(range(1, 10), 3))
mu3_formulas = {(i, j, k): mu3(i, j, k) for (i, j, k) in triples}

print("Precomputing symbolic kappa4(i,j,k,l) for the 70 quadruples actually used "
      "(indices restricted to X, W, WX, Y, WY)...")
quads = list(combinations_with_replacement(NONZERO_GRADIENT_INDICES, 4))
kappa4_formulas = {q: kappa4(*q) for q in quads}
print()

_symbol_to_abc = {sym: key for key, sym in _CANONICAL_RAW_MOMENTS.items() if isinstance(sym, sp.Symbol)}


def _lambdify_all(formulas_dict):
    out = {}
    for key, formula in formulas_dict.items():
        syms = sorted(formula.free_symbols, key=str)
        fn = sp.lambdify(syms, formula, modules="numpy")
        out[key] = (syms, fn)
    return out


print("Lambdifying all formulas (one-time cost)...")
sigma_fns = _lambdify_all(sigma_formulas)
mu3_fns = _lambdify_all(mu3_formulas)
kappa4_fns = _lambdify_all(kappa4_formulas)
print()

def _sigma_stat(raw, i, j):
    return np.mean(raw[i] * raw[j]) - np.mean(raw[i]) * np.mean(raw[j])


def _mu3_stat(raw, i, j, k):
    Vi, Vj, Vk = raw[i], raw[j], raw[k]
    E = np.mean
    return (E(Vi * Vj * Vk) - E(Vi) * E(Vj * Vk) - E(Vj) * E(Vi * Vk)
            - E(Vk) * E(Vi * Vj) + 2 * E(Vi) * E(Vj) * E(Vk))


def _kappa4_stat(raw, i, j, k, l):
    Vi, Vj, Vk, Vl = raw[i], raw[j], raw[k], raw[l]
    E = np.mean
    EA, EB, EC, ED = E(Vi), E(Vj), E(Vk), E(Vl)
    EAB, EAC, EAD = E(Vi * Vj), E(Vi * Vk), E(Vi * Vl)
    EBC, EBD, ECD = E(Vj * Vk), E(Vj * Vl), E(Vk * Vl)
    EABC, EABD = E(Vi * Vj * Vk), E(Vi * Vj * Vl)
    EACD, EBCD = E(Vi * Vk * Vl), E(Vj * Vk * Vl)
    EABCD = E(Vi * Vj * Vk * Vl)
    return (EABCD - EA * EBCD - EB * EACD - EC * EABD - ED * EABC
            - EAB * ECD - EAC * EBD - EAD * EBC
            + 2 * (EA * EB * ECD + EA * EC * EBD + EA * ED * EBC
                   + EB * EC * EAD + EB * ED * EAC + EC * ED * EAB)
            - 6 * EA * EB * EC * ED)


def _batch_means_se(stat_fn, raw, n, n_batches, *idx):
    """Standard error of stat_fn(raw, *idx) computed on the full sample,
    estimated by splitting into n_batches and using the spread of the
    per-batch estimates: SE_full ~= std(batch estimates) / sqrt(n_batches)."""
    batch_size = n // n_batches
    ests = []
    for b in range(n_batches):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        batch_raw = {k: v[sl] for k, v in raw.items()}
        ests.append(stat_fn(batch_raw, *idx))
    return np.std(ests, ddof=1) / np.sqrt(n_batches)


def _check(name, mc_val, formula_val, se, log):
    z = abs(mc_val - formula_val) / se if se > 0 else (0.0 if abs(mc_val - formula_val) < 1e-9 else np.inf)
    ok = z <= Z_THRESHOLD
    if not ok:
        log.append(f"  [FAIL] {name}: MC={mc_val:.5f} formula={formula_val:.5f} SE={se:.5f} z={z:.1f}")
    return ok


n_pass, n_fail = 0, 0

for dgp_name, dgp_fn in DGPS.items():
    print(f"=== DGP: {dgp_name} ===")
    rng = np.random.default_rng(hash(dgp_name) % (2**31))
    X, W, Y = dgp_fn(rng, N)
    raw = {1: X, 2: X ** 2, 3: W, 4: W * X, 5: Y, 6: X * Y, 7: W * Y, 8: W * X ** 2, 9: W * X * Y}

    moment_value = {}
    for sym, (a, b, c) in _symbol_to_abc.items():
        moment_value[sym] = np.mean((X ** a) * (W ** min(b, 1)) * (Y ** c))

    fails = []
    checks = 0

    for (i, j), (syms, fn) in sigma_fns.items():
        if any(s not in moment_value for s in syms):
            continue
        mc_val = _sigma_stat(raw, i, j)
        formula_val = float(fn(*[moment_value[s] for s in syms]))
        se = _batch_means_se(_sigma_stat, raw, N, N_BATCHES, i, j)
        checks += 1
        n_pass += _check(f"sigma({i},{j})", mc_val, formula_val, se, fails)

    for (i, j, k), (syms, fn) in mu3_fns.items():
        if any(s not in moment_value for s in syms):
            continue
        mc_val = _mu3_stat(raw, i, j, k)
        formula_val = float(fn(*[moment_value[s] for s in syms]))
        se = _batch_means_se(_mu3_stat, raw, N, N_BATCHES, i, j, k)
        checks += 1
        n_pass += _check(f"mu3({i},{j},{k})", mc_val, formula_val, se, fails)

    for q, (syms, fn) in kappa4_fns.items():
        if any(s not in moment_value for s in syms):
            continue
        mc_val = _kappa4_stat(raw, *q)
        formula_val = float(fn(*[moment_value[s] for s in syms]))
        se = _batch_means_se(_kappa4_stat, raw, N, N_BATCHES, *q)
        checks += 1
        n_pass += _check(f"kappa4{q}", mc_val, formula_val, se, fails)

    n_this_fail = len(fails)
    n_fail += n_this_fail
    for line in fails:
        print(line)
    print(f"  {checks} checks, {n_this_fail} mismatches (|z| > {Z_THRESHOLD})")
    print()

print(f"TOTAL: {n_pass} passed, {n_fail} failed across {len(DGPS)} DGPs")
if n_fail:
    raise SystemExit(1)
