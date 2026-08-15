"""
Symbolic derivation of the second-order (O(1/n)) term in the finite-sample
MSE of the REG and IREG regression-adjustment estimators.

This reproduces Lemma 11 (REG) and Lemma 12 (IREG) of

    Kelley, A. and Wager, S. "Higher order asymptotics of regression
    adjustment estimators" (2026).

using Hall's (1992, Sec. 2.4) Edgeworth-expansion machinery for smooth
functions of sample means: if tau_hat = g(V_bar) for a vector of sample
moments V_bar with population mean mu, then

    E[n (tau_hat - tau*)^2] = V* + DELTA / n + O(n^{-3/2})

where V* is the usual first-order asymptotic variance and DELTA is a sum
of three terms built from the gradient (a), Hessian (H), and third
derivatives (A3) of g at mu, contracted against the third cumulants and
covariances of the moment vector (both defined once, in moments.py):

    DELTA = T1 + T2 + T3
    T1 = sum_i sum_jk  a_i H_jk mu3(i,j,k)                     (paper's TermB)
    T2 = (1/4) sum_ijkl H_ij H_kl (Sig_ik Sig_jl + Sig_il Sig_jk + Sig_ij Sig_kl)  (TermC)
    T3 = (1/3) sum_ijkl a_i A3_jkl (Sig_ij Sig_kl + Sig_ik Sig_jl + Sig_il Sig_jk) (TermD)

REG uses the 7-moment vector  (X, X^2, W, WX, Y, XY, WY).
IREG extends this with        (X^2 W, XWY).

Both estimators are then rational functions of these sample moments
(closed-form via partitioned-regression algebra for REG, and via the
within-arm-regression representation tau_hat_IREG = tau_hat +
X_bar^T gamma_hat for IREG), differentiated symbolically and evaluated at
the population moment vector mu.

Usage
-----
    python -m src.mse_expansion REG
    python -m src.mse_expansion IREG

or within file:

    from src.mse_expansion import compute_mse_expansion
    result = compute_mse_expansion("IREG")
    print(result.factored_total)
"""

from dataclasses import dataclass
import sys

import sympy as sp

from src.moments import (
    VX, EX3, EX4, EY, EY2, EXY, EX2Y, EX3Y, EWY, EWY2, EXWY, EX2WY, EX3WY,
    EXY2, EX2Y2, EXWY2, EX2WY2,
    sigma, mu3,
)


@dataclass
class MSEExpansionResult:
    model: str                       # "REG" or "IREG"
    gradient: dict                   # {index (1-based): a_i}, nonzero only
    hessian: dict                    # {(i, j): H_ij}, nonzero only
    total: sp.Expr                   # T1 + T2 + T3 (population-moment substituted, expanded)
    factored_total: sp.Expr          # sp.factor(total)


def _partitioned_regression_reg(mvars):
    """tau_hat_REG as a rational function of the 7 sample moments
    (Xbar, X2bar, Wbar, WXbar, Ybar, XYbar, WYbar), via standard
    partitioned-regression algebra (paper eq. after 'Setup for REG')."""
    m1, m2, m3, m4, m5, m6, m7 = mvars
    D = (m2 - m1 ** 2) * (m3 - m3 ** 2) - (m4 - m1 * m3) ** 2
    Num = (m2 - m1 ** 2) * (m7 - m3 * m5) - (m4 - m1 * m3) * (m6 - m1 * m5)
    return sp.cancel(Num / D)


def _within_arm_regression_ireg(mvars):
    """tau_hat_IREG = (alpha1_hat - alpha0_hat) + Xbar*(beta1_hat - beta0_hat),
    the difference of the two within-treatment-arm OLS regressions of Y on
    X (Wager 2025; paper's 'Setup for IREG'), as a rational function of the
    9 sample moments (Xbar, X2bar, Wbar, WXbar, Ybar, XYbar, WYbar, X2Wbar,
    XWYbar)."""
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = mvars
    p1, p0 = m3, 1 - m3
    EX1, EX21, EY1, EXY1 = m4 / p1, m8 / p1, m7 / p1, m9 / p1
    EX0 = (m1 - m4) / p0
    EX20 = (m2 - m8) / p0
    EY0 = (m5 - m7) / p0
    EXY0 = (m6 - m9) / p0
    beta1 = (EXY1 - EX1 * EY1) / (EX21 - EX1 ** 2)
    beta0 = (EXY0 - EX0 * EY0) / (EX20 - EX0 ** 2)
    alpha1 = EY1 - beta1 * EX1
    alpha0 = EY0 - beta0 * EX0
    return sp.cancel((alpha1 - alpha0) + m1 * (beta1 - beta0))


def get_gradient_hessian(model: str):
    """Recompute (dense) gradient a and Hessian H at the population point,
    for model in {"REG","IREG"}. Also used by coverage_cumulants.py,
    which needs the raw arrays rather than the assembled MSE total."""
    P = 7 if model == "REG" else 9
    mvars = list(sp.symbols(f"m1:{P + 1}"))
    A = (_partitioned_regression_reg(mvars) if model == "REG"
         else _within_arm_regression_ireg(mvars))
    pop = {
        mvars[0]: 0, mvars[1]: VX, mvars[2]: sp.Rational(1, 2), mvars[3]: 0,
        mvars[4]: EY, mvars[5]: EXY, mvars[6]: EWY,
    }
    if model == "IREG":
        pop[mvars[7]] = VX / 2
        pop[mvars[8]] = EXWY
    grad = [sp.cancel(sp.diff(A, v)) for v in mvars]
    a = [sp.cancel(g.subs(pop)) for g in grad]
    H_raw = [[sp.cancel(sp.diff(grad[i], mvars[j])) for j in range(P)] for i in range(P)]
    H = [[sp.cancel(H_raw[i][j].subs(pop)) for j in range(P)] for i in range(P)]
    return a, H, P, mvars, grad, H_raw, pop


def compute_mse_expansion(model: str) -> MSEExpansionResult:
    """Compute the O(1/n) term of n*MSE(tau_hat) for model in {"REG","IREG"},
    i.e. reproduce Lemma 11 (REG) or Lemma 12 (IREG)."""
    if model not in ("REG", "IREG"):
        raise ValueError("model must be 'REG' or 'IREG'")

    a, H, P, mvars, grad, H_raw, pop = get_gradient_hessian(model)

    A3 = [[[sp.cancel(sp.diff(H_raw[i][j], mvars[k]).subs(pop)) for k in range(P)]
           for j in range(P)] for i in range(P)]

    # ---- Hall (1992) Term B / TermC / TermD (this module's T1/T2/T3) ----
    T1 = sp.Integer(0)
    for i in range(P):
        if a[i] == 0:
            continue
        for j in range(P):
            for k in range(P):
                if H[j][k] == 0:
                    continue
                T1 += a[i] * H[j][k] * mu3(i + 1, j + 1, k + 1)

    T2 = sp.Integer(0)
    for i in range(P):
        for j in range(P):
            if H[i][j] == 0:
                continue
            for k in range(P):
                for l in range(P):
                    if H[k][l] == 0:
                        continue
                    wick = (sigma(i + 1, k + 1) * sigma(j + 1, l + 1)
                            + sigma(i + 1, l + 1) * sigma(j + 1, k + 1)
                            + sigma(i + 1, j + 1) * sigma(k + 1, l + 1))
                    T2 += sp.Rational(1, 4) * H[i][j] * H[k][l] * wick

    T3 = sp.Integer(0)
    for i in range(P):
        if a[i] == 0:
            continue
        for j in range(P):
            for k in range(P):
                for l in range(P):
                    if A3[j][k][l] == 0:
                        continue
                    wick = (sigma(i + 1, j + 1) * sigma(k + 1, l + 1)
                            + sigma(i + 1, k + 1) * sigma(j + 1, l + 1)
                            + sigma(i + 1, l + 1) * sigma(j + 1, k + 1))
                    T3 += sp.Rational(1, 3) * a[i] * A3[j][k][l] * wick

    total = sp.cancel(sp.expand(T1 + T2 + T3))

    gradient = {i + 1: val for i, val in enumerate(a) if val != 0}
    hessian = {(i + 1, j + 1): H[i][j] for i in range(P) for j in range(P) if H[i][j] != 0}

    return MSEExpansionResult(
        model=model,
        gradient=gradient,
        hessian=hessian,
        total=total,
        factored_total=sp.factor(total),
    )


def _main():
    model = sys.argv[1] if len(sys.argv) > 1 else "IREG"
    result = compute_mse_expansion(model)
    print(f"MODEL = {result.model}")
    print("\nNonzero gradient entries a_i = d(tau_hat)/d(m_i) at mu:")
    for i, val in sorted(result.gradient.items()):
        print(f"  a_{i} = {sp.factor(val)}")
    print("\nO(1/n) coefficient in n*MSE(tau_hat)  [Lemma 11 / Lemma 12]:")
    print(result.factored_total)


if __name__ == "__main__":
    _main()
