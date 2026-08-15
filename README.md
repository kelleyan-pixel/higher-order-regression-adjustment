# Higher-order asymptotics of regression adjustment estimators — code

SymPy code accompanying:

> Anna Kelley and Stefan Wager. "Higher order asymptotics of regression
> adjustment estimators." 2026.

This repository contains the symbolic derivation engines used to compute
every closed-form result in the paper for `X ∈ R`: the second-order
(`O(1/n)`) term in the finite-sample MSE of the REG (ANCOVA) and IREG
(ANCOVA II) regression-adjustment estimators (**Lemma 11**, **Lemma 12**,
**Theorem 1**), and the Edgeworth cumulants behind the oracle CI
coverage result (**Theorem 10**, Appendix A.8), including the full
closed form for the kurtosis cumulant κ₄,₁ referenced there.

## What's here

- `src/moments.py` — the single source of truth for every population
  moment identity used elsewhere in this repository: the covariance
  `sigma(i,j)`, third joint cumulant `mu3(i,j,k)`, and fourth joint
  cumulant `kappa4(i,j,k,l)` of the moment-vector components
  `(X, X², W, WX, Y, XY, WY, X²W, XWY)`. These are pinned down purely
  by `W ∈ {0,1}` and `E[X] = 0` — no regression-model assumptions —
  and computed generically via the standard moment-cumulant relations
  rather than via hand-typed lookup tables.
- `src/mse_expansion.py` — the MSE derivation engine. Given
  `model ∈ {"REG", "IREG"}`, it:
  1. Writes `tau_hat` as a rational function of the relevant sample
     moments (7 moments for REG, 9 for IREG),
  2. Differentiates it symbolically to get the gradient, Hessian, and
     third derivatives at the population moment vector,
  3. Contracts these against `sigma`/`mu3` from `moments.py`, following
     Hall (1992, §2.4)'s expansion for smooth functions of sample means,
  4. Produces a closed-form expression in interpretable population
     moments — Lemma 11 (REG) / Lemma 12 (IREG) / Theorem 1 (the
     difference).
- `src/coverage_cumulants.py` — the Edgeworth cumulants κ₁,₂, κ₃,₁, and
  κ₄,₁ behind Theorem 10, reusing the gradient/Hessian machinery from
  `mse_expansion.py` and the moment identities from `moments.py`.
- `examples/verify_mse.py` — checks the MSE engine against Lemma 11,
  eq. (13), and Lemma 12.
- `examples/verify_coverage_cumulants.py` — checks the symbolic
  assembly of κ₁,₂, κ₃,₁, and κ₄,₁ against eq. (T1R)/(T1I)/(k12)/(k31)
  and the Δ(nκ₄,₁) closed form in Appendix A.8.
- `examples/verify_moments.py` — Monte Carlo checks every `sigma`,
  `mu3`, and `kappa4` identity from `moments.py` (900 checks total)
  against four different simulated `(X, W, Y)` distributions (normal,
  skewed, heavy-tailed, lognormal-misspecified), using batch-means
  standard errors rather than a fixed tolerance (some identities are
  exactly 0 — e.g. a third cumulant with `X` independent of `W` — which
  a naive relative tolerance handles poorly).

## Quick start

```bash
pip install -r requirements.txt
python examples/verify_mse.py
python examples/verify_moments.py
python examples/verify_coverage_cumulants.py
```

or programmatically:

```python
from src.mse_expansion import compute_mse_expansion
from src.coverage_cumulants import compute_kappa41_diff

result = compute_mse_expansion("IREG")
print(result.factored_total)

print(compute_kappa41_diff())   # Delta(n*kappa_4,1), Appendix A.8
```

Runtime: seconds for `verify_mse.py`; a couple of minutes for
`verify_moments.py` (four DGPs × 2M draws each, plus ~900 batch-means
standard errors); several minutes for `verify_coverage_cumulants.py`,
since computing κ₄,₁ contracts the full Hessian against fourth-order
joint cumulants for both estimators.

## Notation: SymPy symbols ↔ paper notation

| Symbol | Paper | Definition |
|---|---|---|
| `VX` | $V_X$ | $\mathbb E[X^2]$ |
| `EX3` | $\kappa_3$ | $\mathbb E[X^3]$ |
| `EX4` | $\kappa_4$ | $\mathbb E[X^4]$ |
| `EY`, `EY2`, `EY3` | $\mathbb E[Y]$, $\mathbb E[Y^2]$, $\mathbb E[Y^3]$ | moments of $Y$ |
| `EXY`, `EX2Y`, `EX3Y` | $\mathbb E[XY]$, $\mathbb E[X^2Y]$, $\mathbb E[X^3Y]$ | |
| `EWY`, `EWY2`, `EWY3` | $\mathbb E[WY]$, $\mathbb E[WY^2]$, $\mathbb E[WY^3]$ | |
| `EXWY`, `EX2WY`, `EX3WY` | $\mathbb E[XWY]$, $\mathbb E[X^2WY]$, $\mathbb E[X^3WY]$ | |
| `EXY2`, `EX2Y2` | $\mathbb E[XY^2]$, $\mathbb E[X^2Y^2]$ | |
| `EXWY2`, `EX2WY2` | $\mathbb E[XWY^2]$, $\mathbb E[X^2WY^2]$ | |
| `M`, `N`, `P` | $M$, $N$, $P$ | $\mathbb E[X^2\varepsilon]$, $\mathbb E[X\varepsilon^2]$, $\mathbb E[X^2(W-\tfrac12)\varepsilon]$ |
| `Q` | $Q$ | $\mathbb E[WX\varepsilon^2]$, the arm-conditional analogue of $N$ (enters only in κ₄,₁; see Appendix A.8) |

`moments.py` uses `_symbol_for(a,b,c)` to name any `E[X^a W^b Y^c]`
consistently, falling back to the canonical names above whenever they
exist. `M, N, P, Q` and the HW-identity substitutions that relate them
to these raw moments (Appendix A.2 Steps 1–3, extended for κ₃,₁ and
κ₄,₁) live in `HW_SUBSTITUTIONS` in `src/coverage_cumulants.py`.

Moment-vector indices `1..7` (REG) / `1..9` (IREG) follow the paper's
ordering: `(X, X², W, WX, Y, XY, WY)`, extended for IREG by `(X²W, XWY)`.

## Requirements

See `requirements.txt`: SymPy and NumPy (the latter only for
`examples/verify_moments.py`).

## Citation

If you use this code, please cite the paper:

```bibtex
@unpublished{kelley2026higher,
  title = {Higher order asymptotics of regression adjustment estimators},
  author = {Kelley, Anna and Wager, Stefan},
  year = {2026}
}
```
