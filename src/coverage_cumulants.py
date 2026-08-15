"""
Symbolic derivation of the Edgeworth cumulants (kappa_1,2, kappa_3,1,
kappa_4,1) used in the proof of the "Oracle CI coverage difference"
theorem (Appendix A.8 of the paper), building on the gradient/Hessian
machinery in mse_expansion.py and the moment identities in moments.py.

Background
----------
Hall (1992, Thm 2.2)'s two-sided coverage expansion needs, for each
estimator tau_hat_j, the cumulants of S_n^j / sqrt(V*):

    kappa_{1,2}^j : O(n^{-1}) term of the 1st cumulant (bias)
    kappa_{2,2}^j : O(n^{-1}) term of the 2nd cumulant (variance)
    kappa_{3,1}^j : O(n^{-1/2}) term of the 3rd cumulant (skewness)
    kappa_{4,1}^j : O(n^{-1}) term of the 4th cumulant (kurtosis)

kappa_{2,2} needs no separate derivation: it combines with kappa_{1,2}^2
to reproduce DIFF/V* = d*, which Theorem 1 already establishes.

kappa_{1,2}: bias cumulant
---------------------------
    kappa_{1,2}^j = sum_kl H^j_kl Sigma_kl / (2 sqrt(V*))

reuses the Hessian H and the covariance identity sigma() from moments.py.

kappa_{3,1}: skewness cumulant
--------------------------------
    (V*)^{3/2} kappa_{3,1}^j = sum_{i,j,k} a_i a_j a_k mu3(i,j,k)
                              + 3 sum_{i,j,k,l} a_i a_j H_kl sigma(i,k) sigma(j,l)

Since a_i is only nonzero at indices {X, W, WX, Y, WY} (1,3,4,5,7), the
first sum only needs mu3 for i,j,k in that set.

kappa_{4,1}: kurtosis cumulant
--------------------------------
    n kappa_{4,1}^j = sum_ijkl a_i a_j a_k a_l kappa4(i,j,k,l)
      + 2 sum_{ijk,pq} a_i a_j a_k H_pq ( 9 terms pairing {i,j,k,p,q}
                                          into one sigma-pair and one
                                          mu3-triple )
      - 6 sum_{i,pq} a_i H_pq mu3(i,p,q)
      - 6 T_1 (a^T Sigma H Sigma a)

The last term vanishes identically (a^T Sigma H Sigma a = 0 for both
REG and IREG), regardless of how T_1 is read.

HW identities
-------------
After substituting the Huber-White identities (Appendix A.2, Steps 1-3,
extended for kappa_3,1 and kappa_4,1 -- see HW_SUBSTITUTIONS),
Delta(n kappa_4,1) = kappa_4,1^IREG - kappa_4,1^REG requires one
population moment beyond M, N, P: Q = E[WX*eps^2], the arm-conditional
analogue of N = E[X*eps^2]. compute_kappa41_diff() returns the closed
form in terms of kappa3, kappa4, V_X, beta*, gamma*, tau*, M, N, P, Q,
and the raw moments E[Y^2], E[X^3 Y], E[X^3 WY] (left unsubstituted, as
in the paper).
"""

import sympy as sp

from src.mse_expansion import get_gradient_hessian
from src.moments import (
    VX, EX3, EY, EWY, EXY, EXWY, EX2Y, EX2WY, EXY2, EXWY2,
    sigma, mu3, kappa4, NONZERO_GRADIENT_INDICES,
)


# ----------------------------------------------------------------
# kappa_1,2
# ----------------------------------------------------------------
def compute_kappa12_raw(model: str) -> sp.Expr:
    """Return sum_kl H_kl * sigma(k,l), i.e. 2*sqrt(V*)*kappa_{1,2}^model."""
    a, H, P, *_ = get_gradient_hessian(model)
    total = sp.Integer(0)
    for k in range(1, P + 1):
        for l in range(1, P + 1):
            if H[k - 1][l - 1] == 0:
                continue
            total += H[k - 1][l - 1] * sigma(k, l)
    return sp.cancel(sp.expand(total))


# ----------------------------------------------------------------
# kappa_3,1
# ----------------------------------------------------------------
def compute_kappa31_raw(model: str) -> sp.Expr:
    """Return (V*)^{3/2} * kappa_{3,1}^model, in raw population moments."""
    a, H, P, *_ = get_gradient_hessian(model)
    idx = NONZERO_GRADIENT_INDICES

    total = sp.Integer(0)
    for i in idx:
        for j in idx:
            for k in idx:
                total += a[i - 1] * a[j - 1] * a[k - 1] * mu3(i, j, k)

    term2 = sp.Integer(0)
    for i in idx:
        for j in idx:
            for k in range(1, P + 1):
                for l in range(1, P + 1):
                    if H[k - 1][l - 1] == 0:
                        continue
                    term2 += a[i - 1] * a[j - 1] * H[k - 1][l - 1] * sigma(i, k) * sigma(j, l)
    total += 3 * term2
    return sp.cancel(sp.expand(total))


# ----------------------------------------------------------------
# kappa_4,1
# ----------------------------------------------------------------
def compute_nkappa41_terms(model: str):
    """Return (term1, term2, term3, term4) of n*kappa_{4,1}^model, per
    Appendix A.8's formula. term4 (the a^T Sigma H Sigma a piece) is
    verified to be identically 0 for both models regardless of the T_1
    prefactor, so it is returned as 0."""
    a, H, P, *_ = get_gradient_hessian(model)
    idx = NONZERO_GRADIENT_INDICES
    nonzero_H = [(p, q) for p in range(1, P + 1) for q in range(1, P + 1) if H[p - 1][q - 1] != 0]

    t1 = sp.Integer(0)
    for i in idx:
        for j in idx:
            for k in idx:
                for l in idx:
                    t1 += a[i - 1] * a[j - 1] * a[k - 1] * a[l - 1] * kappa4(i, j, k, l)
    t1 = sp.cancel(sp.expand(t1))

    t2 = sp.Integer(0)
    for i in idx:
        for j in idx:
            for k in idx:
                aijk = a[i - 1] * a[j - 1] * a[k - 1]
                if aijk == 0:
                    continue
                for (p, q) in nonzero_H:
                    Hpq = H[p - 1][q - 1]
                    inner = (
                        sigma(i, j) * mu3(k, p, q) + sigma(i, k) * mu3(j, p, q)
                        + sigma(j, k) * mu3(i, p, q)
                        + sigma(i, p) * mu3(j, k, q) + sigma(i, q) * mu3(j, k, p)
                        + sigma(j, p) * mu3(i, k, q) + sigma(j, q) * mu3(i, k, p)
                        + sigma(k, p) * mu3(i, j, q) + sigma(k, q) * mu3(i, j, p)
                    )
                    t2 += aijk * Hpq * inner
    t2 = sp.cancel(sp.expand(2 * t2))

    t3 = sp.Integer(0)
    for i in idx:
        if a[i - 1] == 0:
            continue
        for (p, q) in nonzero_H:
            t3 += a[i - 1] * H[p - 1][q - 1] * mu3(i, p, q)
    t3 = sp.cancel(sp.expand(-6 * t3))

    return t1, t2, t3, sp.Integer(0)


def compute_nkappa41_raw(model: str) -> sp.Expr:
    """n * kappa_{4,1}^model, in raw population moments (sum of the four
    terms in Appendix A.8)."""
    t1, t2, t3, t4 = compute_nkappa41_terms(model)
    return sp.cancel(sp.expand(t1 + t2 + t3 + t4))


# ----------------------------------------------------------------
# Huber-White identity substitutions (Appendix A.2 Steps 1-3, extended
# for kappa_3,1 and kappa_4,1), shared by both.
# ----------------------------------------------------------------
gamma, beta, tau, M, N, P, Q = sp.symbols("gamma beta tau M N P Q")
_bp = beta + gamma / 2  # pooled OLS slope

HW_SUBSTITUTIONS = {
    EY: tau / 2, EWY: tau / 2,
    EXY: _bp * VX,
    EXWY: sp.Rational(1, 2) * (_bp + gamma / 2) * VX,
    EX2Y: VX * (tau / 2) + _bp * EX3 + M,
    EX2WY: VX * (tau / 2) + (_bp / 2 + gamma / 4) * EX3 + P + M / 2,
    EXY2: _bp ** 2 * EX3 + (gamma ** 2 / 4) * EX3 + _bp * tau * VX + (tau * gamma / 2) * VX
          + 2 * _bp * M + 2 * gamma * P + N,
    # E[XWY^2] = bp^2 kappa3/2 + bp*(M + 2P + VX*tau + gamma*kappa3)
    #          + gamma^2 kappa3/2 + gamma*(M + 2P + VX*tau) + Q
    EXWY2: _bp ** 2 * EX3 / 2 + _bp * (M + 2 * P + VX * tau + gamma * EX3)
           + gamma ** 2 * EX3 / 2 + gamma * (M + 2 * P + VX * tau) + Q,
}


def compute_kappa41_diff() -> sp.Expr:
    """Delta(n kappa_4,1) = n*kappa_4,1^IREG - n*kappa_4,1^REG, fully
    substituted via HW_SUBSTITUTIONS."""
    reg = compute_nkappa41_raw("REG")
    ireg = compute_nkappa41_raw("IREG")
    diff = sp.expand(ireg - reg)
    return sp.expand(diff.subs(HW_SUBSTITUTIONS))


if __name__ == "__main__":
    for model in ("REG", "IREG"):
        print(f"===== {model} =====")
        print("2*sqrt(V*)*kappa_1,2 =", sp.factor(compute_kappa12_raw(model)))
        print()
    print("Delta(n*kappa_4,1) (fully substituted):")
    print(compute_kappa41_diff())
