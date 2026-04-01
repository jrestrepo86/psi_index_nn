# spec_gradient_stabilization.md — Gradient Stabilization for PSI Neural Estimator

## 1. Problem Statement

The `psi_jef` and `psi_remine` loss functions contain exponential terms (`e^T(x_q)` and `e^{-T(x_p)}`) that are numerically unbounded. During training, the critic network $T_\theta$ is pushed to maximize the gap between P and Q samples. When:

- The learning rate is too high, causing oscillations that push T into extreme values
- P and Q have low overlap (severe drift), so the optimal $T^* = \ln(P/Q)$ is legitimately large
- The network has enough capacity to memorize individual samples

...the exponential terms overflow, the loss diverges, and training collapses.

### Root Cause: Variance of the NWJ Estimator

The NWJ bound (which `psi_jef` is based on) trades the gradient **bias** problem of the Donsker-Varadhan representation for a **variance** problem. The variance of $e^{T^*(x)}$ under Q grows exponentially with the KL divergence being estimated:

$$\text{Var}_Q[e^{T^*}] \propto e^{D_{KL}(P \| Q)}$$

For large PSI values (well-separated distributions), this variance becomes catastrophic even at the theoretical optimum. A single sample where $T(x_q) = 30$ produces $e^{30} \approx 10^{13}$, obliterating the entire batch mean.

### Why `psi_jef` Avoids the DV Bias But Not the Variance

The Donsker-Varadhan bound has $\log(\mathbb{E}_Q[e^T])$, whose minibatch gradient is biased because $\mathbb{E}[A/B] \neq \mathbb{E}[A]/\mathbb{E}[B]$. The MINE paper (Belghazi et al., 2018) fixes this with an EMA on the denominator.

Our `psi_jef` uses the NWJ/f-divergence form where both terms are plain expectations — no ratio, no bias. But the raw $e^T$ term has unbounded variance, which is the problem we address here.

---

## 2. Stabilization Strategy (Layered)

Apply all four layers. Each targets a different failure mode:

| Layer | Mechanism | Targets | Default |
|---|---|---|---|
| 1 | Critic output clamping | Exponential overflow | `clamp_max=10.0` |
| 2 | Weight decay (existing) | Unbounded weight growth | `1e-3` (no change) |
| 3 | Gradient norm clipping | Single-batch gradient spikes | `max_grad_norm=5.0` |
| 4 | `psi_remine` (existing) | Log-partition drift | `remine_reg_weight=0.1` |

---

## 3. Changes to `source/psi_models.py`

### 3.1 Add `clamp_max` Parameter to `PsiModel.__init__`

Add a new constructor parameter `clamp_max: float = 10.0` that controls the symmetric clamping range $[-C, C]$ for the critic output.

**Current signature:**
```python
class PsiModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = [64, 32, 16],
        afn: str = "gelu",
        loss_type: str = "psi_jef",
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
    ):
```

**New signature:**
```python
class PsiModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = [64, 32, 16],
        afn: str = "gelu",
        loss_type: str = "psi_jef",
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
        clamp_max: float = 10.0,
    ):
```

Store as `self.clamp_max = clamp_max`.

### 3.2 Apply Clamping in `forward()`

After computing `t_p` and `t_q` from the network, clamp both before computing the loss terms.

**Current code (lines ~42-44):**
```python
def forward(
    self, data_p: torch.Tensor, data_q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    t_p = self.network(data_p)  # (N, 1)
    t_q = self.network(data_q)  # (N, 1)
```

**New code:**
```python
def forward(
    self, data_p: torch.Tensor, data_q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    t_p = self.network(data_p)  # (N, 1)
    t_q = self.network(data_q)  # (N, 1)

    # Clamp critic outputs to prevent exp overflow
    if self.clamp_max is not None:
        t_p = torch.clamp(t_p, -self.clamp_max, self.clamp_max)
        t_q = torch.clamp(t_q, -self.clamp_max, self.clamp_max)
```

**Rationale:** With `clamp_max=10.0`, the maximum value of $e^T$ is $e^{10} \approx 22{,}026$. This covers pointwise density ratios $P(x)/Q(x)$ up to ~22,000, which is sufficient for any practical data drift scenario. The bias introduced by clamping is negligible for PSI values below $\approx 2C = 20$.

Setting `clamp_max=None` disables clamping entirely for debugging or theoretical experiments.

### 3.3 No Changes to Loss Computation Logic

The `psi_jef` and `psi_remine` loss computation blocks remain identical. Clamping is applied *before* those blocks, so the downstream math is unchanged.

---

## 4. Changes to `source/psi.py`

### 4.1 Pass `clamp_max` Through `Psi.__init__`

Add `clamp_max: float = 10.0` to the `Psi` constructor and forward it to `PsiModel`.

**Current constructor signature:**
```python
class Psi:
    def __init__(
        self,
        X_p,
        X_q,
        hidden_layers: list[int] = [64, 32],
        afn: str = "gelu",
        loss_type: str = "psi_jef",
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
        device: str | None = None,
    ):
```

**New constructor signature:**
```python
class Psi:
    def __init__(
        self,
        X_p,
        X_q,
        hidden_layers: list[int] = [64, 32],
        afn: str = "gelu",
        loss_type: str = "psi_jef",
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
        clamp_max: float = 10.0,
        device: str | None = None,
    ):
```

**Current PsiModel instantiation:**
```python
self.model = PsiModel(
    input_dim=input_dim,
    hidden_layers=hidden_layers,
    afn=afn,
    loss_type=loss_type,
    remine_reg_weight=remine_reg_weight,
    remine_target_val=remine_target_val,
).to(self.device)
```

**New PsiModel instantiation:**
```python
self.model = PsiModel(
    input_dim=input_dim,
    hidden_layers=hidden_layers,
    afn=afn,
    loss_type=loss_type,
    remine_reg_weight=remine_reg_weight,
    remine_target_val=remine_target_val,
    clamp_max=clamp_max,
).to(self.device)
```

### 4.2 Add Gradient Clipping to Training Loop

Add `max_grad_norm: float = 5.0` parameter to `Psi.train()` and apply gradient clipping after `loss.backward()` and before `optimizer.step()`.

**Current `train()` signature:**
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
) -> None:
```

**New `train()` signature:**
```python
def train(
    self,
    batch_size: int = 64,
    num_epochs: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    max_grad_norm: float = 5.0,
    test_size: float = 0.3,
    contiguous_split: bool = False,
    stop_patience: int = 100,
    stop_min_delta: float = 1e-4,
    stop_warmup_steps: int = 1000,
    log_metrics: bool = True,
) -> None:
```

**Current training step (inside epoch loop):**
```python
self.optimizer.zero_grad()
_, loss = self.model(batch_p, batch_q)
loss.backward()
self.optimizer.step()
```

**New training step:**
```python
self.optimizer.zero_grad()
_, loss = self.model(batch_p, batch_q)
loss.backward()
if max_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
self.optimizer.step()
```

Setting `max_grad_norm=None` disables gradient clipping.

### 4.3 Restore Best State Tracking (Currently Commented Out)

The current `psi.py` has best-state tracking commented out:

```python
# if smoothed_loss < best_loss:
#     best_loss = smoothed_loss
#     best_state = copy.deepcopy(self.model.state_dict())
```

**Uncomment this block.** Without it, early stopping fires but restores `None`, meaning the model uses its final (possibly degraded) state. The best-state checkpoint is the most numerically stable point in training.

**Restored code:**
```python
if smoothed_loss < best_loss:
    best_loss = smoothed_loss
    best_state = copy.deepcopy(self.model.state_dict())
```

This requires `import copy` which is already present in `psi.py`.

---

## 5. Changes to `scripts/test_gaussian.py`

### 5.1 Add `clamp_max` to `MODEL_KWARGS`

```python
MODEL_KWARGS = dict(
    hidden_layers=[64, 64, 64, 64],
    afn="relu",
    remine_reg_weight=0.1,
    remine_target_val=0.0,
    clamp_max=10.0,               # NEW
)
```

### 5.2 Add `max_grad_norm` to `TRAIN_KWARGS`

```python
TRAIN_KWARGS = dict(
    num_epochs=5000,
    batch_size=128,
    lr=1e-4,
    weight_decay=1e-5,
    max_grad_norm=5.0,            # NEW
    test_size=0.3,
    contiguous_split=False,
    stop_patience=100,
    stop_min_delta=1e-4,
    stop_warmup_steps=1000,
)
```

---

## 6. Changes to `CLAUDE.md`

Add the following section after "Regularization Design":

```markdown
## Gradient Stabilization

Three mechanisms prevent numerical overflow of the exponential terms in the loss:

1. **Critic output clamping** (`clamp_max=10.0`): Symmetric clamp on T(x) before
   computing exp(T) and exp(-T). Bounds the maximum density ratio to ~22,000.
   Set `clamp_max=None` to disable. Introduces negligible bias for PSI < 20.

2. **Gradient norm clipping** (`max_grad_norm=5.0`): Applied after `loss.backward()`
   and before `optimizer.step()`. Prevents a single bad batch from destroying
   learned weights. Set `max_grad_norm=None` to disable.

3. **Weight decay** (`weight_decay=1e-3`): Existing L2 regularization on network
   weights. Primary smooth regularizer that encourages bounded critic outputs
   indirectly through weight magnitude control.

For severe distribution separation, combine all three with `loss_type="psi_remine"`.
```

---

## 7. Theoretical Justification for Clamping

The SMILE estimator (Song & Ermon, 2020) formalizes critic clamping for mutual information estimation. The key result: clipping $T$ to $[-\tau, \tau]$ yields a biased but dramatically lower-variance estimator. The bias vanishes as $\tau \to \infty$, and for finite $\tau$, the estimator lower-bounds a *clipped* version of the divergence:

$$\hat{D}_\tau = \mathbb{E}_P[\text{clip}(T, -\tau, \tau)] - \log\mathbb{E}_Q[e^{\text{clip}(T, -\tau, \tau)}]$$

For our NWJ-based PSI, the same principle applies. The clamp introduces bias only when the true pointwise log-density ratio $|\ln(P(x)/Q(x))| > \tau$ at some points. With $\tau = 10$:

- Covers density ratios up to $e^{10} \approx 22{,}026$ in either direction
- PSI estimates up to ~20 are essentially unaffected
- In practice, PSI > 10 already indicates catastrophic drift where exact estimation is less important than detecting the problem

---

## 8. Default Rationale

| Parameter | Default | Why |
|---|---|---|
| `clamp_max` | `10.0` | $e^{10} \approx 22{,}000$ covers all practical density ratios. PSI < 20 unaffected. Matches SMILE paper recommendation range. |
| `max_grad_norm` | `5.0` | Standard value for transformer/MLP training. Prevents catastrophic updates while allowing normal gradient flow. |
| `weight_decay` | `1e-3` | Unchanged. Already tuned for PSI stability in Gaussian validation. |

---

## 9. Verification

After implementing, run the existing validation:

```bash
python scripts/test_gaussian.py
```

**Expected behavior changes:**

- Scenario (a) No Shift: should remain PASS, potentially faster convergence
- Scenario (b) Mean Shift: should remain PASS with tighter variance across runs
- Scenario (c) Covariance Shift: should PASS more reliably; this is where instability was most likely

**Additional manual test — stress test with high learning rate:**

```python
# This should no longer diverge
psi = Psi(X_p, X_q, loss_type="psi_jef", clamp_max=10.0)
psi.train(lr=1e-3, weight_decay=1e-3, max_grad_norm=5.0, num_epochs=2000)
print(psi.get_psi())
```

Without clamping and gradient clipping, `lr=1e-3` frequently causes loss divergence. With stabilization, it should converge (possibly to a slightly less accurate estimate due to the aggressive learning rate, but without NaN or inf).

---

## 10. Files Modified (Summary)

| File | Changes |
|---|---|
| `source/psi_models.py` | Add `clamp_max` param; apply `torch.clamp` in `forward()` |
| `source/psi.py` | Pass `clamp_max` to `PsiModel`; add `max_grad_norm` to `train()`; apply `clip_grad_norm_`; uncomment best-state tracking |
| `scripts/test_gaussian.py` | Add `clamp_max` and `max_grad_norm` to kwargs dicts |
| `CLAUDE.md` | Add "Gradient Stabilization" documentation section |

No changes to `psi_sampler.py`, `utils.py`, `gaussian_psi.py`, or any frozen MINE files.
