"""
Verify the kappa_1,2, kappa_3,1, and kappa_4,1 coverage cumulants
against the paper's stated equations (Appendix A.8). Run with:

    python examples/verify_coverage_cumulants.py

This checks the symbolic assembly of each cumulant from the gradient,
Hessian, and moment identities:
  1. kappa_1,2 (both REG and IREG) against eq. (T1R)/(T1I) and eq. (k12).
  2. kappa_3,1 against eq. (k31), confirming REG == IREG.
  3. Delta(n*kappa_4,1) against the closed form in Appendix A.8 (the
     a^T Sigma H Sigma a term vanishes identically; the remainder is a
     function of kappa3, kappa4, V_X, beta*, gamma*, tau*, M, N, P, and
     Q = E[WX*eps^2]).

The underlying moment identities (sigma, mu3, kappa4 from src/moments.py)
are Monte Carlo checked separately, across several DGPs, in
examples/verify_moments.py.

Step 3 is the slow part (a few minutes) since it involves contracting
the full Hessian against fourth-order joint cumulants for both models.
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage_cumulants import (
    compute_kappa12_raw, compute_kappa31_raw, compute_kappa41_diff,
    HW_SUBSTITUTIONS,
)
from src.moments import VX, EX3, EX4, EY, EWY, EXY, EXWY, EX2Y, EX2WY, EXY2, EXWY2, EY2, EWY2

PASS, FAIL = [], []


def check(name, expr):
    ok = sp.simplify(expr) == 0
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"       residual: {sp.simplify(expr)}")


# ---------------------------------------------------------------------
# kappa_1,2 vs. paper's eq. (T1R)/(T1I) and eq. (k12)
# ---------------------------------------------------------------------
print("--- kappa_1,2 (bias cumulant) ---")
raw_reg = compute_kappa12_raw("REG")
raw_ireg = compute_kappa12_raw("IREG")

paper_T1R = 4 * (2 * EWY * VX - 2 * EX2WY + EX2Y - EY * VX) / VX
paper_T1I = 4 * (2 * EWY * VX ** 2 - 2 * EX2WY * VX + EX2Y * VX
                  + 2 * EX3 * EXWY - EX3 * EXY - EY * VX ** 2) / VX ** 2
check("kappa_1,2 raw sum (REG) == paper eq. (T1R)", raw_reg - paper_T1R)
check("kappa_1,2 raw sum (IREG) == paper eq. (T1I)", raw_ireg - paper_T1I)

gamma, beta, tau, M, N, P, Q = sp.symbols("gamma beta tau M N P Q")
raw_reg_sub = sp.simplify(raw_reg.subs(HW_SUBSTITUTIONS))
raw_ireg_sub = sp.simplify(raw_ireg.subs(HW_SUBSTITUTIONS))
paper_2sqrtVstar_k12_R = -2 * (EX3 * gamma + 4 * P) / VX
paper_2sqrtVstar_k12_I = -8 * P / VX
check("kappa_1,2 (REG, substituted) == paper eq. (k12)", raw_reg_sub - paper_2sqrtVstar_k12_R)
check("kappa_1,2 (IREG, substituted) == paper eq. (k12)", raw_ireg_sub - paper_2sqrtVstar_k12_I)
print()

# ---------------------------------------------------------------------
# kappa_3,1 vs. paper's eq. (k31)
# ---------------------------------------------------------------------
print("--- kappa_3,1 (skewness cumulant) ---")
k31_reg = compute_kappa31_raw("REG")
k31_ireg = compute_kappa31_raw("IREG")
check("kappa_3,1: REG == IREG", k31_reg - k31_ireg)

EY3, EWY3 = sp.symbols("EY3 EWY3")
# eq. (k31) deliberately leaves EXY2, EXWY2, EWY2, EWY3, EY3 as raw moments
# (only Y, WY, XY, XWY, X^2Y, X^2WY are substituted) -- use a matching
# partial substitution here rather than the full HW_SUBSTITUTIONS dict.
k31_substitutions = {k: v for k, v in HW_SUBSTITUTIONS.items() if k not in (EXY2, EXWY2)}
k31_sub = sp.expand(k31_reg.subs(k31_substitutions))

paper_eq_k31 = -4 * (
    6 * EWY2 * tau - 4 * EWY3 - 3 * EX3 * beta ** 2 * gamma
    + 12 * EXWY2 * beta - 6 * EXY2 * beta + 2 * EY3
    - 12 * P * beta ** 2 - 6 * VX * beta ** 2 * tau - 3 * VX * beta * gamma * tau - 2 * tau ** 3
    + 6 * EXWY2 * gamma - 3 * EXY2 * gamma - 12 * P * beta * gamma - 6 * VX * beta * gamma * tau
    - 3 * EX3 * beta * gamma ** 2 - 3 * P * gamma ** 2 - 3 * VX * gamma ** 2 * tau
    - sp.Rational(3, 4) * EX3 * gamma ** 3
)
check("kappa_3,1 (substituted) == paper eq. (k31)", k31_sub - paper_eq_k31)
print()

# ---------------------------------------------------------------------
# Delta(n*kappa_4,1) vs. Appendix A.8's closed form
# ---------------------------------------------------------------------
print("--- kappa_4,1 (kurtosis cumulant) ---")
print("(slow -- contracting the full Hessian against 4th-order cumulants for both models)")
computed_diff = compute_kappa41_diff()

EX3Y, EX3WY = sp.symbols("EX3Y EX3WY")
paper_inner = (
    4 * EX3 * EY2 * tau - 8 * EX3 * M * beta * gamma - 10 * EX3 * M * gamma ** 2
    - 4 * EX3 * VX * beta ** 2 * tau - 4 * EX3 * VX * beta * gamma * tau - EX3 * VX * gamma ** 2 * tau
    - 2 * EX3 * tau ** 3 - EX3 * tau
    - 16 * EX3WY * EY2 + 16 * EX3WY * VX * beta ** 2 + 16 * EX3WY * VX * beta * gamma
    + 4 * EX3WY * VX * gamma ** 2 + 8 * EX3WY * tau ** 2 + 4 * EX3WY
    + 8 * EX3Y * EY2 - 8 * EX3Y * VX * beta ** 2 - 8 * EX3Y * VX * beta * gamma
    - 2 * EX3Y * VX * gamma ** 2 - 4 * EX3Y * tau ** 2 - 2 * EX3Y
    - 4 * EY2 * VX ** 2 * gamma - 16 * M ** 2 * gamma + 8 * M * N - 16 * M * P * gamma - 16 * M * Q
    - 8 * M * VX * gamma * tau + 4 * VX ** 3 * beta ** 2 * gamma + 4 * VX ** 3 * beta * gamma ** 2
    + VX ** 3 * gamma ** 3 + 2 * VX ** 2 * gamma * tau ** 2 + VX ** 2 * gamma
)
paper_formula = (-12 * gamma / VX) * paper_inner
check("Delta(n*kappa_4,1) == Appendix A.8 closed form", computed_diff - paper_formula)

print()
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    raise SystemExit(1)
