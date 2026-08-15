"""
Population moments for the (X, W, Y) joint distribution: the single
source of truth for every symbol and identity used elsewhere in this
repository (mse_expansion.py, coverage_cumulants.py).

Everything here follows from three facts about the underlying random
variables, and nothing else -- no regression-model structure, no
Huber-White decomposition:

    W in {0, 1}, so W^k = W for every k >= 1.
    E[X] = 0.
    (nothing else is assumed about the joint law of (X, W, Y))

Given that, the covariance Sigma(i,j), third joint cumulant mu3(i,j,k),
and fourth joint cumulant kappa4(i,j,k,l) of any of the nine moment-
vector components

    V = (X, X^2, W, WX, Y, XY, WY, X^2 W, XWY)     [indices 1..9]

(the ordering used throughout the paper's appendix) are pinned down
completely by the standard moment-cumulant relations, evaluated on raw
population moments E[X^a W^b Y^c]. raw_E() computes exactly that: it
takes any polynomial in X, W, Y, reduces powers of W via W^2 = W, and
maps what's left to a named population-moment symbol -- either one of
the paper's existing ones (VX, kappa3, M, N, P, ...) or a newly-named
one, generated consistently by _symbol_for.

sigma(), mu3(), and kappa4() below are the ONLY definitions of these
quantities anywhere in this codebase; mse_expansion.py and
coverage_cumulants.py both call them directly rather than keeping their
own copies.
"""

import sympy as sp

X, W, Y = sp.symbols("X W Y")

VX = sp.Symbol("VX", positive=True)
EX3 = sp.Symbol("EX3")
EX4 = sp.Symbol("EX4")
EY = sp.Symbol("EY")
EY2 = sp.Symbol("EY2")
EY3 = sp.Symbol("EY3")
EY4 = sp.Symbol("EY4")
EXY = sp.Symbol("EXY")
EX2Y = sp.Symbol("EX2Y")
EX3Y = sp.Symbol("EX3Y")
EXY2 = sp.Symbol("EXY2")
EX2Y2 = sp.Symbol("EX2Y2")
EXY3 = sp.Symbol("EXY3")
EWY = sp.Symbol("EWY")
EWY2 = sp.Symbol("EWY2")
EWY3 = sp.Symbol("EWY3")
EWY4 = sp.Symbol("EWY4")
EXWY = sp.Symbol("EXWY")
EX2WY = sp.Symbol("EX2WY")
EX3WY = sp.Symbol("EX3WY")
EXWY2 = sp.Symbol("EXWY2")
EX2WY2 = sp.Symbol("EX2WY2")
EXWY3 = sp.Symbol("EXWY3")
EX2WY3 = sp.Symbol("EX2WY3")

# (a, b, c) -> symbol for E[X^a W^b Y^c], b already reduced to {0,1}.
_CANONICAL_RAW_MOMENTS = {
    (0, 0, 0): sp.Integer(1),
    (1, 0, 0): sp.Integer(0), (2, 0, 0): VX, (3, 0, 0): EX3, (4, 0, 0): EX4,
    (0, 1, 0): sp.Rational(1, 2), (1, 1, 0): sp.Integer(0),
    (2, 1, 0): VX / 2, (3, 1, 0): EX3 / 2, (4, 1, 0): EX4 / 2,
    (0, 0, 1): EY, (1, 0, 1): EXY, (2, 0, 1): EX2Y, (3, 0, 1): EX3Y,
    (0, 1, 1): EWY, (1, 1, 1): EXWY, (2, 1, 1): EX2WY, (3, 1, 1): EX3WY,
    (0, 0, 2): EY2, (1, 0, 2): EXY2, (2, 0, 2): EX2Y2,
    (0, 1, 2): EWY2, (1, 1, 2): EXWY2, (2, 1, 2): EX2WY2,
    (0, 0, 3): EY3, (1, 0, 3): EXY3,
    (0, 1, 3): EWY3, (1, 1, 3): EXWY3, (2, 1, 3): EX2WY3,
    (0, 0, 4): EY4, (0, 1, 4): EWY4,
}
_auto_raw_moment_cache = {}


def _symbol_for(a, b, c):
    """Population moment E[X^a W^b Y^c] (b already reduced to 0/1), using
    the canonical names above where available, else a consistently
    auto-generated new symbol."""
    key = (a, b, c)
    if key in _CANONICAL_RAW_MOMENTS:
        return _CANONICAL_RAW_MOMENTS[key]
    if key in _auto_raw_moment_cache:
        return _auto_raw_moment_cache[key]
    name = f"E{'X' + str(a) if a else ''}{'W' if b else ''}{'Y' + str(c) if c else ''}"
    sym = sp.Symbol(name or "E1")
    _auto_raw_moment_cache[key] = sym
    return sym


def reduce_W(expr):
    """W in {0,1} => W^k = W for k >= 1."""
    expr = sp.expand(expr)
    for n in range(12, 1, -1):
        expr = expr.subs(W ** n, W)
    return sp.expand(expr)


def raw_E(expr):
    """Population expectation of a polynomial in X, W, Y."""
    expr = reduce_W(expr)
    total = 0
    for term in sp.Add.make_args(expr):
        coeff, monom = term.as_coeff_Mul()
        a = sp.degree(monom, X) if monom.has(X) else 0
        b = 1 if monom.has(W) else 0
        c = sp.degree(monom, Y) if monom.has(Y) else 0
        total += coeff * _symbol_for(a, b, c)
    return sp.expand(total)

V = {1: X, 2: X ** 2, 3: W, 4: W * X, 5: Y, 6: X * Y, 7: W * Y, 8: W * X ** 2, 9: W * X * Y}


def sigma(i, j):
    """Cov(V_i, V_j)."""
    Vi, Vj = V[i], V[j]
    return sp.cancel(sp.expand(raw_E(Vi * Vj) - raw_E(Vi) * raw_E(Vj)))


def mu3(i, j, k):
    """Third joint cumulant of (V_i, V_j, V_k)."""
    Vi, Vj, Vk = V[i], V[j], V[k]
    EA, EB, EC = raw_E(Vi), raw_E(Vj), raw_E(Vk)
    EAB, EAC, EBC = raw_E(Vi * Vj), raw_E(Vi * Vk), raw_E(Vj * Vk)
    EABC = raw_E(Vi * Vj * Vk)
    return sp.cancel(sp.expand(EABC - EA * EBC - EB * EAC - EC * EAB + 2 * EA * EB * EC))


def kappa4(i, j, k, l):
    """Fourth joint cumulant of (V_i, V_j, V_k, V_l)."""
    Vi, Vj, Vk, Vl = V[i], V[j], V[k], V[l]
    EA, EB, EC, ED = raw_E(Vi), raw_E(Vj), raw_E(Vk), raw_E(Vl)
    EAB, EAC, EAD = raw_E(Vi * Vj), raw_E(Vi * Vk), raw_E(Vi * Vl)
    EBC, EBD, ECD = raw_E(Vj * Vk), raw_E(Vj * Vl), raw_E(Vk * Vl)
    EABC, EABD = raw_E(Vi * Vj * Vk), raw_E(Vi * Vj * Vl)
    EACD, EBCD = raw_E(Vi * Vk * Vl), raw_E(Vj * Vk * Vl)
    EABCD = raw_E(Vi * Vj * Vk * Vl)
    return sp.cancel(sp.expand(
        EABCD
        - EA * EBCD - EB * EACD - EC * EABD - ED * EABC
        - EAB * ECD - EAC * EBD - EAD * EBC
        + 2 * (EA * EB * ECD + EA * EC * EBD + EA * ED * EBC
               + EB * EC * EAD + EB * ED * EAC + EC * ED * EAB)
        - 6 * EA * EB * EC * ED
    ))


# Indices used by both estimators (X, X^2, W, WX, Y, XY, WY) vs. the two
# additional IREG-only moment-vector components (X^2 W, XWY).
REG_INDICES = (1, 2, 3, 4, 5, 6, 7)
IREG_ONLY_INDICES = (8, 9)
IREG_INDICES = REG_INDICES + IREG_ONLY_INDICES

# Indices (matching the moment-vector ordering above) where the
# gradient of tau_hat is nonzero for both REG and IREG: X, W, WX, Y, WY.
NONZERO_GRADIENT_INDICES = (1, 3, 4, 5, 7)


def model_indices(model):
    return REG_INDICES if model == "REG" else IREG_INDICES
