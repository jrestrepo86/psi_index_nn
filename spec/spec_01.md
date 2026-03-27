# spec_01.md — PSI Neural Estimation: Project Specification

## 1. Project Goal and Scope

### Purpose

This project implements **Population Stability Index (PSI) estimation via neural variational bounds**, providing a model-free, distribution-free alternative to the traditional binning-based PSI. The primary use case is **data drift detection** in machine learning production systems where distributions may be continuous, multivariate, and high-dimensional.

### Why Neural PSI?

The classical histogram-based PSI breaks down in multivariate settings due to the curse of dimensionality. This project estimates PSI directly from samples using the Donsker-Varadhan / NWJ variational representation of f-divergences, parameterized by a neural network critic. This enables:

- Continuous, differentiable estimation with no binning artifacts
- Scaling to arbitrary dimensions
- Sensitivity to both mean shift and covariance drift (unlike univariate PSI per-feature)

### Scope

- **In scope**: Multivariate PSI estimation from samples, symmetric Jeffreys divergence, single-critic architecture, weight decay regularization, ReMINE regularization variant, production-ready class API mirroring the MINE pattern, Gaussian ground-truth validation.
- **Out of scope**: Conditional PSI, feature attribution for drift, online/streaming estimation, ONNX export.

---

## 2. Theoretical Foundations

### 2.1 Definition

PSI is the symmetric Kullback-Leibler divergence (Jeffreys divergence) between a reference distribution Q and a current distribution P:

```
PSI(P, Q) = D_KL(P || Q) + D_KL(Q || P)
           = ∫ (P(x) - Q(x)) · ln(P(x) / Q(x)) dx
```

By convention in data monitoring, P is the "actual" (production) distribution and Q is the "expected" (training/reference) distribution. The value is non-negative and equals zero if and only if P = Q.

### 2.2 Variational Lower Bound

Using the Fenchel-Legendre conjugate representation (centered NWJ bound), each KL divergence is lower-bounded by:

```
D_KL(P || Q) >= sup_T [ E_P[T(x)] - E_Q[e^(T(x)) - 1] ]
```

The supremum is attained at the optimal critic T*(x) = ln(P(x)/Q(x)).

### 2.3 The Symmetric Single-Critic Trick

The key insight enabling an efficient implementation: because PSI requires both KL(P||Q) and KL(Q||P), one would naively need two critics T1 and T2. However, the optimal critics satisfy:

```
T2*(x) = ln(Q(x)/P(x)) = -ln(P(x)/Q(x)) = -T1*(x)
```

Substituting T2 = -T with T1 = T, the full variational lower bound for PSI becomes:

```
PSI(P, Q) >= sup_T [ E_P[T(x) - e^(-T(x)) + 1] - E_Q[e^(T(x)) + T(x) - 1] ]
```

This single-critic formulation requires exactly one network to estimate the full symmetric divergence.

### 2.4 Training Objective

Since PyTorch optimizers minimize, the loss function negates the variational bound:

```
L(θ) = -( mean_{x~P}[T_θ(x) - e^(-T_θ(x)) + 1] - mean_{x~Q}[e^(T_θ(x)) + T_θ(x) - 1] )
```

The estimated PSI at convergence is `-L(θ_optimal)`.

### 2.5 Exact Gaussian Formula (Validation Ground Truth)

For multivariate Gaussians P = N(μ_P, Σ_P) and Q = N(μ_Q, Σ_Q), the closed-form PSI is:

```
PSI = 0.5 * [ tr(Σ_Q⁻¹ Σ_P + Σ_P⁻¹ Σ_Q) - 2k + (μ_P - μ_Q)ᵀ (Σ_P⁻¹ + Σ_Q⁻¹) (μ_P - μ_Q) ]
```

where k is the dimensionality. Implemented in `source/gaussian_psi.py` — serves as ground truth for all validation experiments.

### 2.6 Regularization Theory

When the supports of P and Q are well-separated (severe drift), T_θ(x) → ±∞ and the exponential terms cause numerical blow-up. Two defenses are required:

1. **Weight Decay (L2 regularization)**: Penalizes large weight magnitudes, encouraging smooth quadratic surfaces needed for covariance-shift estimation. This is the primary regularization mechanism — stronger weight decay (1e-3) replaces the need for explicit gradient penalties.

2. **Early Stopping**: Monitors smoothed PSI estimate, stops when improvement plateaus, preventing overfitting to sample-level separation artifacts.

---

## 3. Architecture Design

### 3.1 File Structure

```
psi_index/
├── README.md                        (existing: theory documentation)
├── spec/
│   └── spec_01.md                   (this document)
├── source/
│   ├── batch_sampler.py             (existing: MineSampler, reused as pattern reference)
│   ├── gaussian_psi.py              (existing: exact formula, reused as validation oracle)
│   ├── mine.py                      (existing: MINE reference implementation — do not modify)
│   ├── models.py                    (existing: MINE Model — do not modify)
│   ├── psi.py                       (NEW: main Psi class, analogous to mine.py)
│   ├── psi_models.py                (NEW: PsiModel nn.Module, analogous to models.py)
│   ├── psi_sampler.py               (NEW: PsiSampler, analogous to batch_sampler.py)
│   ├── psi_model.py                 (DEPRECATED: v1 prototype — reference only)
│   └── psi_model_02.py              (DEPRECATED: v2 prototype — reference only)
└── tests/
    ├── test_psi_gaussian.py         (NEW: Gaussian validation tests — slow)
    └── test_psi_unit.py             (NEW: unit tests — fast)
```

### 3.2 Parallel Structure: MINE vs PSI

| MINE | PSI | Role |
|---|---|---|
| `source/mine.py` | `source/psi.py` | Orchestrator class |
| `source/models.py` | `source/psi_models.py` | nn.Module critic |
| `source/batch_sampler.py` | `source/psi_sampler.py` | Data sampling |
| `Mine` class | `Psi` class | Public API |
| `Model` class | `PsiModel` class | Neural network |
| `get_mi()` | `get_psi()` | Estimation method |
| `loss_type`: mine/nwj/remine | `loss_type`: psi_jef/psi_remine | Loss variant |

---

## 4. API Design

### 4.1 Class `Psi` — Constructor

```python
class Psi:
    def __init__(
        self,
        X_p,                             # Samples from P (actual/production), array-like
        X_q,                             # Samples from Q (reference/expected), array-like
        hidden_layers: list[int] = [64, 32],
        afn: str = "gelu",
        loss_type: str = "psi_jef",
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
        device: str | None = None,
    )
```

- `X_p` and `X_q` passed through `to_col_vector()` (same pattern as `Mine.__init__`)
- `input_dim` inferred as `X_p.shape[1]` after column vectorization
- PSI critic operates on the feature space directly — not on concatenated pairs like MINE
- `remine_reg_weight` and `remine_target_val` are passed through to `PsiModel`; ignored when `loss_type="psi_jef"`
- `device` defaults to CUDA if available, else CPU

### 4.2 Method `train()`

```python
def train(
    self,
    batch_size: int = 64,
    num_epochs: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    test_size: float = 0.3,
    contiguous_split: bool = False,
    stop_patience: int = 100,
    stop_min_delta: float = 1e-4,
    stop_warmup_steps: int = 1000,
    log_metrics: bool = True,
) -> None
```

**Default differences from Mine.train():**
- `lr = 1e-4` (vs Mine's `1e-5`): PSI loss landscape is more stable
- `weight_decay = 1e-3` (vs Mine's `5e-5`): Stronger regularization for numerical stability

### 4.3 Method `get_psi()`

```python
def get_psi(self) -> float
```

- Raises `ValueError` if `not self.trained`
- Uses the full dataset (no subsampling)
- Returns `max(0.0, estimate)` — clamps to non-negative

### 4.4 Method `plot_metrics()`

```python
def plot_metrics(self, text: str = "", show: bool = True) -> go.Figure
```

- 2-row Plotly subplot: row 1 = Loss + Smoothed Loss; row 2 = PSI estimate over epochs
- Mirrors `Mine.plot_metrics()` exactly

### 4.5 Class `PsiSampler`

```python
class PsiSampler:
    def __init__(self, X_p: np.ndarray, X_q: np.ndarray, rng_seed: int | None = None)
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (batch_from_P, batch_from_Q), shape (batch_size, d) each
        # Sampling with replacement (handles batch_size > n)
        # Returns CPU float32 tensors, requires_grad=False
```

**Do not reuse `MineSampler`** — MineSampler concatenates X and Y for joint/marginal pairs. PSI requires independently drawing from two distinct distributions.

---

## 5. Loss Variants

### 5.1 `"psi_jef"` — Jeffreys (symmetric NWJ, default)

```python
term_p = mean(t_p - exp(-t_p) + 1)   # E_P[T - e^{-T} + 1]
term_q = mean(exp(t_q) + t_q - 1)    # E_Q[e^T + T - 1]
psi_est = term_p - term_q
loss = -psi_est
```

The `+1` / `-1` centering constants ensure PSI=0 when P=Q, reducing variance at the optimum. Weight decay is the sole regularizer in this variant.

### 5.2 `"psi_remine"` — Jeffreys with ReMINE Regularization

Adapts the ReMINE regularizer (from `models.py`) to the symmetric PSI objective. The regularizer penalizes deviation of the log-partition term from a target value, reducing variance in early training when the exponential terms in `term_q` are volatile:

```python
term_p = mean(t_p - exp(-t_p) + 1)           # E_P[T - e^{-T} + 1]
log_partition = logsumexp(t_q, 0) - log(N)   # log E_Q[e^T], stabilized
term_q_remine = exp(log_partition) + mean(t_q) - 1
psi_est = term_p - term_q_remine
base_loss = -psi_est
reg = remine_reg_weight * (log_partition - remine_target_val) ** 2
loss = base_loss + reg
```

**Important**: `psi_est` is derived from `base_loss` only — the regularization term does not inflate the estimate.

The `log_partition` term uses `logsumexp` for numerical stability (same pattern as `models.py` line 90). `remine_target_val=0.0` is the correct default because when P=Q, the optimal critic T*(x)=0 and `log E_Q[e^0]=0`.

### 5.3 Rejected Variants

- **Separate two-critic**: Theoretically valid but wasteful — single-critic trick is equivalent at half the parameters.
- **MINE-style EMA**: MINE's EMA stabilizes the log-partition function for KL. PSI's `psi_jef` loss has no log-partition function (exponentials computed directly), so EMA does not apply. `psi_remine` uses `logsumexp` instead for the same stabilization effect.
- **Gradient penalty (WGAN-GP)**: Replaced by weight decay — L2 regularization on network weights provides sufficient Lipschitz control without requiring interpolated samples or second-order gradients.

---

## 6. Key Implementation Decisions

### 6.1 Activation Function: GELU as Default

PSI critic approximates `ln(P/Q)`, a smooth function. GELU provides smoother gradients than ReLU, empirically improving convergence for covariance-shift scenarios where the optimal critic is a quadratic form `xᵀ A x`. The `afn` parameter accepts any string supported by `get_activation_fn` (`"relu"`, `"elu"`, `"gelu"`, `"tanh"`).

### 6.2 Optimizer: Schedule-Free AdamW

Use `schedulefree.AdamWScheduleFree` identically to `mine.py`. Requires explicit `optimizer.train()` / `optimizer.eval()` mode toggling at the correct points in the training loop (same as `mine.py` lines 124, 136).

### 6.3 Weight Decay Default: 1e-3

Significantly stronger than MINE's `5e-5`. Without strong L2 regularization, the PSI critic can grow without bound under well-separated distributions, causing `e^T` and `e^{-T}` to overflow.

### 6.4 Best State Restoration

Track the epoch with the lowest smoothed test loss. Save `deepcopy(model.state_dict())` at that epoch. Restore on early stopping or end of training. This ensures `get_psi()` uses the most numerically stable checkpoint.

### 6.5 Independent Train/Test Splits for P and Q

P and Q may have different sizes and are split independently using `train_test_split`. The `batch_size` is applied to each independently.

### 6.6 Device Management

- Sampler always returns CPU tensors (`requires_grad=False`)
- `.to(self.device)` called in the training loop (same pattern as `mine.py` lines 121-122)

### 6.7 PSI Non-Negativity Clamping

`get_psi()` returns `max(0.0, estimate.item())`. Raw training metrics remain unclamped so convergence behavior is visible in `plot_metrics()`.

### 6.8 Hidden Layer Default: `[64, 32]` (Pyramid)

MINE uses `[64, 64]` (flat). PSI uses a pyramid `[64, 32]` — fewer parameters reduces overfitting of the log-density ratio to finite samples.

---

## 7. Validation Approach

### 7.1 Ground Truth

`source/gaussian_psi.py`'s `exact_gaussian_psi(mu_p, cov_p, mu_q, cov_q)` provides analytically exact PSI for Gaussian distributions. All validation tests compare neural estimates against this.

### 7.2 Validation Scenarios

| Scenario | Setup | Exact PSI | Tolerance |
|---|---|---|---|
| No drift | P = Q = N(0, I₂) | 0.0 | abs error < 0.1 |
| Mean shift | P=N([0,0],I), Q=N([1.5,1.5],I) | ~4.50 | rel error < 10% |
| Covariance shift | P=N(0,Σ_+0.8), Q=N(0,Σ_-0.8) | ~3.55 | rel error < 15% |
| Combined drift | P=N(0,Σ_+0.8), Q=N([1,-1],I) | exact_gaussian_psi | rel error < 15% |

Covariance shift is the hardest case — requires strong weight decay (≥1e-3) and smooth activation (GELU). Use `"psi_remine"` if `"psi_jef"` underestimates.

### 7.3 Validation Helper

```python
def validate_gaussian_psi(
    mu_p, cov_p, mu_q, cov_q,
    n_samples=5000,
    rtol=0.10,
    **train_kwargs
) -> dict:
    # Returns {"exact": float, "estimated": float, "rel_error": float, "passed": bool}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (`tests/test_psi_unit.py`) — Fast

| Test | What is checked |
|---|---|
| Sampler shapes | `PsiSampler.sample()` returns tensors of shape `(batch_size, d)` |
| Sampler independence | The two returned tensors are not identical |
| Model forward shapes | `PsiModel.forward()` returns `(scalar, scalar)` for both loss types |
| Loss differentiability | `loss.backward()` succeeds for both `"psi_jef"` and `"psi_remine"` |
| Constructor | `Psi` instantiates for 1D and 5D inputs |
| Untrained guard | `get_psi()` raises `ValueError` before `train()` |
| Training runs | `train(num_epochs=10)` completes without exception |
| Trained flag | `self.trained` is True after `train()` |
| Non-negativity | `get_psi()` >= 0 when P = Q |
| plot_metrics | Returns a `go.Figure` |

### 8.2 Validation Tests (`tests/test_psi_gaussian.py`) — Slow (`pytest.mark.slow`)

One test per scenario from section 7.2. Use `torch.manual_seed(42)` and `np.random.seed(42)`. Use `n_samples=5000`.

### 8.3 Test Fixtures

- `torch.manual_seed(42)` and `np.random.seed(42)` for reproducibility
- `n_samples=1000` for unit tests (speed), `n_samples=5000` for validation tests (accuracy)

---

## 9. `Psi.train()` Sequence Diagram

```
Psi.train()
│
├── Create AdamWScheduleFree(params, lr, weight_decay, warmup_steps=1000)
├── Create EarlyStopping(patience, delta, warmup_steps)
├── Create ExpMovingAverageSmooth()
├── train_test_split(P) → train_idx_p, test_idx_p
├── train_test_split(Q) → train_idx_q, test_idx_q
├── PsiSampler(P_train, Q_train) → train_sampler
├── PsiSampler(P_test, Q_test)   → test_sampler
│
└── for epoch in range(num_epochs):
    │
    ├── [TRAIN]
    │   ├── batch_p, batch_q = train_sampler.sample(batch_size)
    │   ├── batch_p, batch_q = .to(device)
    │   ├── model.train(), optimizer.train()
    │   ├── optimizer.zero_grad()
    │   ├── psi_est, loss = model(batch_p, batch_q)
    │   ├── loss.backward()
    │   └── optimizer.step()
    │
    └── [EVAL]
        ├── batch_p, batch_q = test_sampler.sample(batch_size)
        ├── batch_p, batch_q = .to(device)
        ├── model.eval(), optimizer.eval()
        ├── with torch.no_grad(): psi_est, loss = model(batch_p, batch_q)
        ├── smoothed_loss = smooth(loss.item())
        ├── if log_metrics: metrics.append(epoch, loss, smoothed_loss, psi_est)
        ├── if smoothed_loss < best_loss: best_state = deepcopy(state_dict)
        ├── early_stopping(smoothed_loss)
        └── if early_stopping.early_stop: model.load_state_dict(best_state); break
```

---

## 10. Development Guidelines (Basis for CLAUDE.md)

### 10.1 Code Style

- Follow exact patterns in `mine.py` and `models.py` for analogous functionality
- Type hints on all function signatures (Python 3.10+ union syntax `X | Y`)
- Import order: stdlib → third-party (torch, numpy, pandas, plotly, schedulefree) → local (minepy.utils)
- All docstrings in English

### 10.2 Error Handling

- `get_psi()` raises `ValueError("Did you call .train()?")` — same message pattern as `Mine.get_mi()`
- `plot_metrics()` raises `ValueError("No metrics to plot. Did you call .train()?")`
- `PsiModel` raises `ValueError` for unknown `loss_type` — same pattern as `models.py` line 100

### 10.3 No Breaking Changes to Existing Files

- `mine.py`, `models.py`, `batch_sampler.py`, `gaussian_psi.py` must not be modified
- `psi_model.py` and `psi_model_02.py` are deprecated but not deleted (implementation reference)
- New code lives exclusively in `psi.py`, `psi_models.py`, `psi_sampler.py`, and `tests/`

### 10.4 Dependency Boundaries

```
psi.py          → psi_models.py, psi_sampler.py, minepy.utils.utils, schedulefree, torch, numpy, pandas, plotly
psi_models.py   → minepy.utils.utils (get_activation_fn), torch
psi_sampler.py  → numpy, torch
```

No circular imports. `gaussian_psi.py` remains standalone.

---

## 11. Known Issues in Prototypes (to Fix in New Implementation)

| Issue | Location | Description | Fix |
|---|---|---|---|
| Full-batch training | both prototypes | Full dataset as single batch, does not scale | Use `batch_size` minibatches with sampler |
| No train/test split | both prototypes | No validation data, dishonest early stopping | Split both P and Q independently |
| Early stopping direction | `psi_model_02.py:95` | Tracks `best_psi` (maximize) inconsistently | Standardize on smoothed test loss (minimize) |

---

## 12. Hyperparameter Recommendations

| Scenario | `loss_type` | `weight_decay` | `hidden_layers` |
|---|---|---|---|
| Low-dim, mild drift | `"psi_jef"` | 1e-3 | [64, 32] |
| High-dim features | `"psi_jef"` | 1e-3 | [128, 64, 32] |
| Pure covariance shift | `"psi_remine"` | 1e-3 | [64, 32] |
| Severe separation | `"psi_remine"` | 5e-3 | [64, 32] |
| Large samples (>50k) | `"psi_jef"` | 1e-4 | [64, 32] |
