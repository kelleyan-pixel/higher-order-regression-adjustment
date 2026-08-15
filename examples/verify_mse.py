"""
Verify the MSE engine (src/mse_expansion.py) against Lemma 11, eq. (13),
and Lemma 12 of the paper. Run with:

    python examples/verify_mse.py
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mse_expansion import compute_mse_expansion
from src.moments import VX, EX3, EX4, EY, EXY, EWY, EX2Y, EX2WY, EX3Y, EXWY

PASS, FAIL = [], []


def check(name, expr):
    ok = sp.simplify(expr) == 0
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"       residual: {sp.simplify(expr)}")


reg = compute_mse_expansion("REG")
ireg = compute_mse_expansion("IREG")

# --- Lemma 11 ---------------------------------------------------------
EY2, EX2Y2, EXY2 = sp.symbols("EY2 EX2Y2 EXY2")
lemma11 = (
    -96 / VX * EWY * EX2WY + 48 / VX * EWY * EX2Y + 64 / VX**2 * EX2WY**2
    - 64 / VX**2 * EX2WY * EX2Y + 48 / VX * EX2WY * EY + 16 / VX**2 * EX2Y**2
    - 16 / VX * EX2Y * EY - 4 / VX * EX2Y2 - 8 / VX**2 * EX3 * EXY * EY
    + 8 / VX**2 * EX3Y * EXY - 4 / VX**3 * EX4 * EXY**2
    + 96 / VX * EXWY**2 - 96 / VX * EXWY * EXY + 12 / VX * EXY**2
    - 16 * EY**2 + 12 * EY2
)
check("engine(REG) == paper Lemma 11", reg.total - lemma11)

# --- eq. (13): DIFF = IREG - REG ---------------------------------------
paper_diff_full = (1 / VX**2) * (
    8 * EY**2 * VX**2
    + (-36 * EXY**2 + 144 * EXY * EXWY - 144 * EXWY**2 - 16 * EX2Y * EY) * VX
    + 8 * EX2Y**2 + 8 * EX3 * EXY2 + 32 * EX3 * EWY * (2 * EXWY - EXY)
    + 32 * EX3 * EY * (EXY - EXWY)
    + (1 / VX) * (
        4 * EX4 * EXY**2 + 16 * EX4 * EXWY**2 - 16 * EX4 * EXWY * EXY
        + 64 * EX3 * EX2WY * EXY - 128 * EX3 * EX2WY * EXWY
        + 64 * EX3 * EX2Y * EXWY - 64 * EX3 * EX2Y * EXY
    )
    + (EX3**2 / VX**2) * (24 * EXY**2 - 32 * EXY * EXWY + 32 * EXWY**2)
)
check("engine(IREG) - engine(REG) == paper eq. (13)", (ireg.total - reg.total) - paper_diff_full)

# --- Lemma 12 ---------------------------------------------------------
lemma12 = (
    -96 / VX * EWY * EX2WY + 48 / VX * EWY * EX2Y
    + 64 / VX**2 * EWY * EX3 * EXWY - 32 / VX**2 * EWY * EX3 * EXY
    + 64 / VX**2 * EX2WY**2 - 64 / VX**2 * EX2WY * EX2Y
    - 128 / VX**3 * EX2WY * EX3 * EXWY + 64 / VX**3 * EX2WY * EX3 * EXY
    + 48 / VX * EX2WY * EY + 24 / VX**2 * EX2Y**2
    + 64 / VX**3 * EX2Y * EX3 * EXWY - 64 / VX**3 * EX2Y * EX3 * EXY
    - 32 / VX * EX2Y * EY - 4 / VX * EX2Y2
    + 32 / VX**4 * EX3**2 * EXWY**2 - 32 / VX**4 * EX3**2 * EXWY * EXY
    + 24 / VX**4 * EX3**2 * EXY**2 - 32 / VX**2 * EX3 * EXWY * EY
    + 24 / VX**2 * EX3 * EXY * EY + 8 / VX**2 * EX3 * EXY2
    + 8 / VX**2 * EX3Y * EXY + 16 / VX**3 * EX4 * EXWY**2
    - 16 / VX**3 * EX4 * EXWY * EXY
    - 48 / VX * EXWY**2 + 48 / VX * EXWY * EXY - 24 / VX * EXY**2
    - 8 * EY**2 + 12 * EY2
)
check("engine(IREG) == paper Lemma 12", ireg.total - lemma12)

print()
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    raise SystemExit(1)
