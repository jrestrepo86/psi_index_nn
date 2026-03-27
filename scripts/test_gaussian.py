"""
Standalone Gaussian validation for the PSI neural estimator.

Tests three 2D Gaussian scenarios and compares neural estimates
against the exact closed-form PSI.

Usage:
    python scripts/test_gaussian.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from source.gaussian_psi import exact_gaussian_psi
from source.psi import Psi

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

N_SAMPLES = 5000
TRAIN_KWARGS = dict(num_epochs=500, batch_size=64)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_scenario(name, X_p, X_q, exact, loss_type, abs_tol=None, rel_tol=None):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Exact PSI  : {exact:.4f}")

    psi = Psi(X_p, X_q, loss_type=loss_type)
    psi.train(**TRAIN_KWARGS)
    estimated = psi.get_psi()

    print(f"  Estimated  : {estimated:.4f}")

    if abs_tol is not None:
        err = abs(estimated - exact)
        status = "PASS" if err < abs_tol else "FAIL"
        print(f"  Abs error  : {err:.4f}  (tol={abs_tol})  →  {status}")

    if rel_tol is not None:
        rel_err = abs(estimated - exact) / (abs(exact) + 1e-8)
        status = "PASS" if rel_err < rel_tol else "FAIL"
        print(f"  Rel error  : {rel_err:.4f}  (tol={rel_tol})  →  {status}")

    psi.plot_metrics(text=f"[{name}]", show=False)
    return estimated


# ── (a) No Shift ──────────────────────────────────────────────────────────────
mu    = torch.zeros(2)
cov_i = torch.eye(2)

dist  = MultivariateNormal(mu, cov_i)
X_p   = dist.sample((N_SAMPLES,)).numpy()
X_q   = dist.sample((N_SAMPLES,)).numpy()

exact_a = exact_gaussian_psi(mu, cov_i, mu, cov_i)

run_scenario(
    "(a) No Shift  —  P = Q = N([0,0], I)",
    X_p, X_q,
    exact=exact_a,
    loss_type="psi_jef",
    abs_tol=0.1,
)

# ── (b) Mean Shift ────────────────────────────────────────────────────────────
mu_p  = torch.zeros(2)
mu_q  = torch.tensor([1.5, 1.5])
cov_i = torch.eye(2)

dist_p = MultivariateNormal(mu_p, cov_i)
dist_q = MultivariateNormal(mu_q, cov_i)
X_p    = dist_p.sample((N_SAMPLES,)).numpy()
X_q    = dist_q.sample((N_SAMPLES,)).numpy()

exact_b = exact_gaussian_psi(mu_p, cov_i, mu_q, cov_i)

run_scenario(
    "(b) Mean Shift  —  P=N([0,0],I), Q=N([1.5,1.5],I)",
    X_p, X_q,
    exact=exact_b,
    loss_type="psi_jef",
    rel_tol=0.10,
)

# ── (c) Covariance Shift ──────────────────────────────────────────────────────
mu_zero = torch.zeros(2)
cov_pos = torch.tensor([[1.0,  0.8], [ 0.8, 1.0]])
cov_neg = torch.tensor([[1.0, -0.8], [-0.8, 1.0]])

dist_p = MultivariateNormal(mu_zero, cov_pos)
dist_q = MultivariateNormal(mu_zero, cov_neg)
X_p    = dist_p.sample((N_SAMPLES,)).numpy()
X_q    = dist_q.sample((N_SAMPLES,)).numpy()

exact_c = exact_gaussian_psi(mu_zero, cov_pos, mu_zero, cov_neg)

run_scenario(
    "(c) Covariance Shift  —  P=N(0,Σ+0.8), Q=N(0,Σ-0.8)",
    X_p, X_q,
    exact=exact_c,
    loss_type="psi_remine",
    rel_tol=0.15,
)

print("\nDone.")
