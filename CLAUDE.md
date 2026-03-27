# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Neural estimation of the **Population Stability Index (PSI)** — the symmetric KL divergence (Jeffreys divergence) between a reference distribution Q and a current distribution P — using variational lower bounds parameterized by a neural network critic. Solves the curse of dimensionality that breaks classical histogram-based PSI in multivariate settings.

## Commands

This project is a library under development. There is no build step. Run tests with:

```bash
# All tests
pytest

# Fast unit tests only
pytest tests/test_psi_unit.py

# Slow validation tests (Gaussian ground truth)
pytest tests/test_psi_gaussian.py -m slow

# Single test
pytest tests/test_psi_unit.py::test_name -v
```

Dependencies: `torch`, `numpy`, `pandas`, `plotly`, `schedulefree`, and `minepy` (local package providing `minepy.utils.utils`).

## Architecture

### Parallel Structure: MINE → PSI

The PSI implementation **mirrors** the existing MINE implementation exactly. Always consult the MINE files as the authoritative pattern before writing any PSI code:

| PSI file (new) | MINE file (reference pattern) | Role |
|---|---|---|
| `source/psi.py` | `source/mine.py` | Orchestrator class |
| `source/psi_models.py` | `source/models.py` | `nn.Module` critic |
| `source/psi_sampler.py` | `source/batch_sampler.py` | Data sampling |

### Key Architectural Difference: PSI vs MINE

- **MINE** critic takes *concatenated* `(X, Y)` pairs → `input_dim = dim(X) + dim(Y)`
- **PSI** critic takes samples from a *single* distribution → `input_dim = dim(X_p) = dim(X_q)`
- **PSI sampler** draws independently from two distributions P and Q; do not reuse `MineSampler` (it concatenates for joint/marginal pairs)

### Frozen Files

`mine.py`, `models.py`, `batch_sampler.py`, `gaussian_psi.py` **must not be modified**. `psi_model.py` and `psi_model_02.py` are deprecated prototypes — reference only, do not modify.

### Dependency Boundaries

```
psi.py        → psi_models.py, psi_sampler.py, minepy.utils.utils, schedulefree, torch, numpy, pandas, plotly
psi_models.py → minepy.utils.utils (get_activation_fn), torch
psi_sampler.py → numpy, torch
gaussian_psi.py → standalone (torch only)
```

## Loss Variants

Two loss types, both negating the variational bound so PyTorch minimizers apply:

**`"psi_jef"` (default)** — symmetric NWJ bound:
```
loss = -( mean[T(x_p) - exp(-T(x_p)) + 1] - mean[exp(T(x_q)) + T(x_q) - 1] )
```

**`"psi_remine"`** — same bound with `logsumexp`-stabilized log-partition regularizer (mirrors `models.py:89-94`):
```
log_partition = logsumexp(t_q, 0) - log(N)
loss = base_loss + remine_reg_weight * (log_partition - remine_target_val)^2
```
`psi_est` is always derived from `base_loss` only — the regularizer does not inflate the estimate.

## Regularization Design

Weight decay is the **sole** regularizer for numerical stability (default `1e-3`, much stronger than MINE's `5e-5`). This replaces gradient penalty. Without sufficient weight decay, `T(x) → ±∞` under well-separated distributions causing `exp` overflow.

Use `"psi_remine"` over `"psi_jef"` for pure covariance shift or severe separation scenarios.

## Validation

`source/gaussian_psi.py::exact_gaussian_psi(mu_p, cov_p, mu_q, cov_q)` provides closed-form PSI for Gaussians — used as ground truth for all validation tests. Tolerances: abs error < 0.1 for no-drift case; rel error < 10–15% for drift cases with `n_samples=5000`.

## Code Style

- Follow patterns in `mine.py` / `models.py` exactly for analogous functionality
- Type hints on all signatures using Python 3.10+ union syntax (`X | Y`)
- Import order: stdlib → third-party → local (`minepy.utils`)
- Error messages must match MINE's patterns: `"Did you call .train()?"`, `"No metrics to plot. Did you call .train()?"`
- `PsiModel` raises `ValueError` for unknown `loss_type` — same as `models.py:100`
