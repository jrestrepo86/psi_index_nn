# Build Plan 01: PSI Model Architecture + Gaussian Validation Script

## Context

Build the three core PSI source files (`psi_sampler.py`, `psi_models.py`, `psi.py`) mirroring the MINE architecture, plus a standalone script `scripts/test_gaussian.py` that validates the model on three 2D Gaussian scenarios: no shift, mean shift, and covariance shift.

Utilities live in `source/utils.py`. Import path: `from source.utils import ...`

---

## Files to Create

### 1. `source/psi_sampler.py`

**`PsiSampler`** — mirrors `MineSampler` but draws independently from two separate distributions:

```python
class PsiSampler:
    def __init__(self, X_p: np.ndarray, X_q: np.ndarray, rng_seed: int | None = None):
        self.rng = np.random.default_rng(rng_seed)
        self.x_p = X_p
        self.x_q = X_q
        self.n_p = X_p.shape[0]
        self.n_q = X_q.shape[0]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx_p = self.rng.choice(self.n_p, size=batch_size, replace=True)
        idx_q = self.rng.choice(self.n_q, size=batch_size, replace=True)
        batch_p = torch.tensor(self.x_p[idx_p], dtype=torch.float32, requires_grad=False)
        batch_q = torch.tensor(self.x_q[idx_q], dtype=torch.float32, requires_grad=False)
        return batch_p, batch_q
```

Key difference from `MineSampler`: `replace=True` handles `batch_size > n`; separate `n_p`/`n_q` for unequal dataset sizes; no concatenation.

---

### 2. `source/psi_models.py`

**`PsiModel(nn.Module)`** — mirrors `models.py` `Model` class:

- Network: `Linear(input_dim, h[0]) → act → ... → Linear(h[-1], 1)` (identical construction to `models.py:60-67`)
- `get_activation_fn` imported from `source.utils`

**`forward(data_p, data_q) → tuple[torch.Tensor, torch.Tensor]`** returns `(psi_est, loss)`:

**`"psi_jef"` variant:**
```python
t_p = self.network(data_p)          # (N, 1)
t_q = self.network(data_q)          # (N, 1)
term_p = (t_p - torch.exp(-t_p) + 1).mean()
term_q = (torch.exp(t_q) + t_q - 1).mean()
psi_est = term_p - term_q
loss = -psi_est
return psi_est, loss
```

**`"psi_remine"` variant** (mirrors `models.py:89-94` remine pattern):
```python
t_p = self.network(data_p)
t_q = self.network(data_q)
term_p = (t_p - torch.exp(-t_p) + 1).mean()
log_partition = torch.logsumexp(t_q.squeeze(), dim=0) - math.log(t_q.shape[0])
term_q = torch.exp(log_partition) + t_q.mean() - 1
psi_est = term_p - term_q
base_loss = -psi_est
reg = self.remine_reg_weight * (log_partition - self.remine_target_val) ** 2
loss = base_loss + reg
# psi_est from base_loss only
return psi_est, loss
```

Raises `ValueError` for unknown `loss_type` (same pattern as `models.py:100`).

---

### 3. `source/psi.py`

**`Psi`** — mirrors `mine.py` `Mine` class exactly, with these differences:

| | `Mine` | `Psi` |
|---|---|---|
| Inputs | X, Y (concatenated for critic) | X_p, X_q (each fed to critic separately) |
| `input_dim` | `x.shape[1] + y.shape[1]` | `x_p.shape[1]` |
| Sampler | `MineSampler(x[idx], y[idx])` | `PsiSampler(x_p[idx_p], x_q[idx_q])` — **independent splits** |
| Default `lr` | 1e-5 | 1e-4 |
| Default `weight_decay` | 5e-5 | 1e-3 |
| Metric key | `"mi"` | `"psi"` |
| `get_mi()` | returns raw `mi.item()` | `get_psi()` returns `max(0.0, psi_est.item())` |
| Plot row 2 label | `"MI"` | `"PSI"` |

**Independent P/Q splits** (critical — not in MINE):
```python
train_idx_p, test_idx_p = train_test_split(self.x_p.shape[0], test_size, ...)
train_idx_q, test_idx_q = train_test_split(self.x_q.shape[0], test_size, ...)
train_sampler = PsiSampler(self.x_p[train_idx_p], self.x_q[train_idx_q])
test_sampler  = PsiSampler(self.x_p[test_idx_p],  self.x_q[test_idx_q])
```

Training loop body is identical to `mine.py:118-161` — optimizer `.train()`/`.eval()` toggling, `deepcopy` on best smoothed loss, early stopping.

Import from `source.utils` (not `minepy.utils.utils`).

---

### 4. `scripts/test_gaussian.py`

Standalone script (not pytest). Run with: `python scripts/test_gaussian.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch, numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from source.psi import Psi
from source.gaussian_psi import exact_gaussian_psi
```

**Three scenarios** — seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `n_samples=5000`:

| | Setup | loss_type | Exact PSI | Tolerance |
|---|---|---|---|---|
| (a) No shift | P = Q = N([0,0], I) | `psi_jef` | 0.0 | abs < 0.1 |
| (b) Mean shift | P=N([0,0],I), Q=N([1.5,1.5],I) | `psi_jef` | ~4.50 | rel < 10% |
| (c) Cov shift | P=N(0,Σ+0.8), Q=N(0,Σ-0.8) | `psi_remine` | ~3.55 | rel < 15% |

Covariance matrices: `Σ+0.8 = [[1, 0.8],[0.8, 1]]`, `Σ-0.8 = [[1,-0.8],[-0.8,1]]`.

Each scenario prints:
```
=== (a) No Shift ===
Exact PSI  : 0.0000
Estimated  : X.XXXX
Abs error  : X.XXXX  →  PASS / FAIL
```

Training kwargs for all scenarios: `num_epochs=500, batch_size=64`. The script calls `psi.plot_metrics(show=False)` silently (no display during batch run).

---

## File Structure After Implementation

```
source/
  psi_sampler.py     (NEW)
  psi_models.py      (NEW)
  psi.py             (NEW)
  utils.py           (existing — imported by new files)
  mine.py            (frozen reference)
  models.py          (frozen reference)
  batch_sampler.py   (frozen reference)
  gaussian_psi.py    (frozen reference)
scripts/
  test_gaussian.py   (NEW)
```

---

## Verification

```bash
python scripts/test_gaussian.py
```

Expected output: all three scenarios PASS with PSI estimates within tolerance.
