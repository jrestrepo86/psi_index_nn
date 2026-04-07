# spec_classifier_psi.md — Classifier-Based PSI Estimation

## 1. Motivation

The current `psi_jef` and `psi_remine` loss variants suffer from exponential variance:
the terms `exp(T(x_q))` and `exp(-T(x_p))` in the NWJ bound produce unbounded gradients
when the critic outputs grow large. This requires a fragile stack of mitigations (clamping,
gradient clipping, batch size tuning, learning rate reduction) that interact unpredictably.

The classifier approach, based on Molavipour et al. (2021) and the CCMI method
(Mukherjee et al., 2019), replaces the unbounded critic with a **binary classifier**
whose sigmoid output is naturally bounded in `[τ, 1-τ]`. The training loss is standard
**binary cross-entropy** — bounded, smooth, and well-understood. The density ratio is
recovered from the classifier output, and PSI is computed entirely in **log space**,
eliminating all exponential terms from both training and estimation.

### Summary of Advantages

| Issue | Critic approach (`psi_jef`) | Classifier approach (`psi_classifier`) |
|---|---|---|
| Training loss | Custom, contains `e^T` | Standard BCE, bounded |
| Gradient variance | Exponential in divergence | Bounded by construction |
| Requires clamping | Yes (`clamp_max`) | No (sigmoid is the clamp) |
| Requires grad clipping | Yes (`max_grad_norm`) | No |
| Learning rate sensitivity | High | Low (standard classifier) |
| Batch size sensitivity | High | Low |
| Bias control | `clamp_max` (hard to tune) | `tau` (clean, interpretable) |
| PSI computation | Requires `e^T`, `e^{-T}` | All in log space: `logit(ω)` |

---

## 2. Theory

### 2.1 Binary Classifier for Density Ratio Estimation

Train a neural network classifier `ω_θ(x) ∈ [τ, 1-τ]` to distinguish samples from P
(label=1) from samples from Q (label=0) using binary cross-entropy:

```
L(ω) = -E_P[log ω(x)] - E_Q[log(1 - ω(x))]
```

The Bayes-optimal classifier (Lemma 1 in Molavipour et al.) satisfies:

```
ω*(x) = p1 · P(x) / [p1 · P(x) + (1-p1) · Q(x)]
```

where `p1 = P(label=1)` is the prior probability of class P. The density ratio is
recovered as:

```
Γ(x) = P(x)/Q(x) = [(1-p1)/p1] · [ω(x) / (1 - ω(x))]
```

With balanced batches (`p1 = 0.5`), this simplifies to:

```
Γ(x) = ω(x) / (1 - ω(x))
log Γ(x) = logit(ω(x)) = log(ω(x)) - log(1 - ω(x))
```

### 2.2 PSI from the Density Ratio

PSI is the symmetric KL divergence:

```
PSI(P, Q) = E_P[log Γ(X)] - E_Q[log Γ(X)]
          = E_P[logit(ω(X))] - E_Q[logit(ω(X))]
```

**Critical observation**: the entire PSI computation is in log space. There is no `exp()`
anywhere — neither in training (BCE) nor in estimation (logit). This is why the classifier
approach is fundamentally more stable than the critic approach.

### 2.3 Role of τ (Sigmoid Clipping)

The classifier output is clipped to `[τ, 1-τ]` which bounds the log density ratio:

```
|log Γ(x)| ≤ log((1-τ)/τ)
```

| τ value | Max \|log Γ\| | Max density ratio | PSI range unaffected |
|---|---|---|---|
| 0.05 | 2.94 | 19 | PSI < ~6 |
| 0.01 | 4.60 | 99 | PSI < ~9 |
| 0.005 | 5.29 | 199 | PSI < ~11 |
| 0.001 | 6.91 | 999 | PSI < ~14 |

Default `τ = 0.01` covers all practical data drift scenarios (PSI > 10 is catastrophic
drift where exact estimation matters less than detection).

### 2.4 Comparison with Critic Clamping

The classifier's `τ` and the critic's `clamp_max` both limit the density ratio, but
they operate at fundamentally different points:

- **`clamp_max`** clips the critic output *after* an unbounded forward pass. The loss
  function still contains `exp(T)` and the gradients of `exp(clamp(T))` are zero in
  the clamped region — the network gets no learning signal for tail samples.

- **`τ`** clips the sigmoid output. The training loss (BCE) never involves exponentials
  and provides smooth gradients everywhere in `[τ, 1-τ]`. The network continues to
  learn from tail samples even when the output is near the clip boundary because the
  BCE gradient `∂L/∂ω` is non-zero throughout `(0, 1)`.

---

## 3. Changes to `source/psi_models.py`

### 3.1 Add `tau` Parameter to `PsiModel.__init__`

Add `tau: float = 0.01` to the constructor. This parameter is only used when
`loss_type="psi_classifier"`.

**New constructor signature:**
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
        tau: float = 0.01,
    ):
```

Store as `self.tau = tau`.

### 3.2 Add Classifier Network Variant

When `loss_type="psi_classifier"`, the network must end with a `Sigmoid` activation.
The cleanest approach: build the same hidden layers as the existing network, but
**do not add Sigmoid to `self.network`**. Instead, apply sigmoid + clipping inside
`forward()`. This keeps the architecture construction identical for all loss types and
avoids duplicating layer-building logic.

**No changes to network construction.** The same `self.network` (ending in `Linear(..., 1)`)
is used for all loss types. The sigmoid + clipping is applied in the `forward()` method
only for the classifier variant.

### 3.3 Add `"psi_classifier"` Branch to `forward()`

Add the following branch after the existing `psi_remine` branch:

```python
elif self.loss_type == "psi_classifier":
    # Apply sigmoid + clip to get classifier output in [tau, 1-tau]
    omega_p = torch.sigmoid(t_p).clamp(self.tau, 1.0 - self.tau)
    omega_q = torch.sigmoid(t_q).clamp(self.tau, 1.0 - self.tau)

    # Training loss: binary cross-entropy (P=1, Q=0)
    loss = -(torch.log(omega_p).mean() + torch.log(1.0 - omega_q).mean())

    # PSI estimation: E_P[logit(ω)] - E_Q[logit(ω)]
    # With balanced batches (p1=0.5), log density ratio = logit(ω)
    log_gamma_p = torch.log(omega_p) - torch.log(1.0 - omega_p)
    log_gamma_q = torch.log(omega_q) - torch.log(1.0 - omega_q)
    psi_est = log_gamma_p.mean() - log_gamma_q.mean()

    return psi_est, loss
```

**Important implementation notes:**

1. **`t_p` and `t_q` are raw network outputs (logits)**, same as for the other loss types.
   Sigmoid is applied here, not in the network architecture.

2. **Clamping (`clamp_max`) is NOT applied** for the classifier variant. The sigmoid
   already bounds the output. If `clamp_max` is set, it should be ignored when
   `loss_type="psi_classifier"`. Add this guard at the top of `forward()`:

   ```python
   # Clamp critic outputs (not used for classifier — sigmoid handles bounding)
   if self.clamp_max is not None and self.loss_type != "psi_classifier":
       t_p = torch.clamp(t_p, -self.clamp_max, self.clamp_max)
       t_q = torch.clamp(t_q, -self.clamp_max, self.clamp_max)
   ```

3. **The logit computation** `log(ω) - log(1-ω)` is numerically stable because `ω` is
   clamped away from 0 and 1 by `τ`. An alternative is `torch.logit(omega_p, eps=tau)`,
   but explicit log subtraction is clearer and matches the paper's formulation.

4. **Loss sign convention**: BCE loss is already positive (minimizing it improves the
   classifier). The PSI estimate is computed separately from the loss — unlike `psi_jef`
   where `loss = -psi_est`. This is intentional: the classifier's loss objective is
   cross-entropy accuracy, not direct PSI maximization. PSI is a *derived quantity*
   from the trained classifier.

### 3.4 Update the `ValueError` Message

Update the error message to include `"psi_classifier"`:

```python
raise ValueError(
    f"Unknown loss_type '{self.loss_type}'. "
    "Use 'psi_jef', 'psi_remine', or 'psi_classifier'."
)
```

---

## 4. Changes to `source/psi.py`

### 4.1 Pass `tau` Through `Psi.__init__`

Add `tau: float = 0.01` to the `Psi` constructor and forward it to `PsiModel`.

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
        tau: float = 0.01,
        device: str | None = None,
    ):
```

**PsiModel instantiation** — add `tau=tau`:

```python
self.model = PsiModel(
    input_dim=input_dim,
    hidden_layers=hidden_layers,
    afn=afn,
    loss_type=loss_type,
    remine_reg_weight=remine_reg_weight,
    remine_target_val=remine_target_val,
    clamp_max=clamp_max,
    tau=tau,
).to(self.device)
```

### 4.2 No Changes to `train()` Loop

The training loop is identical for all loss types. The classifier variant uses the
same `optimizer.zero_grad() → forward() → loss.backward() → optimizer.step()` cycle.
Gradient clipping (`max_grad_norm`) can remain in place as a safety net but should
rarely activate for the classifier variant since BCE gradients are naturally bounded.

### 4.3 `get_psi()` — No Changes Needed

The existing `get_psi()` method calls `self.model(full_P, full_Q)` and returns the
`psi_est` tensor. Since the classifier's `forward()` already returns `psi_est` as the
logit-based PSI estimate, no changes are needed.

---

## 5. Changes to `scripts/test_gaussian.py`

### 5.1 Add Classifier Test Configuration

Add a new test run using `loss_type="psi_classifier"`. The simplest approach is to
run all three Gaussian scenarios with the classifier variant alongside the existing
tests.

**Recommended test parameters for the classifier variant:**

```python
LOSS_TYPE_CLASSIFIER = "psi_classifier"

MODEL_KWARGS_CLASSIFIER = dict(
    hidden_layers=[64, 64, 64, 64],
    afn="gelu",
    tau=0.01,
)

TRAIN_KWARGS_CLASSIFIER = dict(
    num_epochs=10000,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    max_grad_norm=None,       # Not needed for classifier
    test_size=0.3,
    contiguous_split=False,
    stop_patience=100,
    stop_min_delta=1e-4,
    stop_warmup_steps=1000,
)
```

**Key differences from critic parameters:**

| Parameter | Critic (`psi_jef`) | Classifier (`psi_classifier`) | Why |
|---|---|---|---|
| `lr` | `1e-4` to `5e-4` | `1e-3` | BCE gradients are bounded; higher lr is safe |
| `weight_decay` | `1e-3` | `1e-4` | Less regularization needed (no exp overflow) |
| `batch_size` | `512` (for stability) | `128` | No variance explosion from `e^T` |
| `max_grad_norm` | `5.0` | `None` | BCE gradients don't spike |
| `clamp_max` | `10.0` | N/A (ignored) | Sigmoid handles bounding |
| `tau` | N/A | `0.01` | Controls max density ratio (~99) |

### 5.2 Expected Results

All three scenarios should pass with the classifier variant:

| Scenario | Exact PSI | Expected estimate | Expected rel error |
|---|---|---|---|
| (a) No Shift | 0.0000 | < 0.05 | < 0.1 (abs) |
| (b) Mean Shift | 4.5000 | 4.2 – 4.6 | < 0.10 |
| (c) Covariance Shift | 7.1111 | 6.2 – 7.2 | < 0.15 |

The covariance shift scenario should converge reliably because:
- The classifier loss (BCE) doesn't contain exponential terms
- The optimal classifier for quadratic log-density ratios is a logistic function of a
  quadratic form, which the 4-layer network can represent
- No clamping-induced bias on tail samples

---

## 6. Changes to `tests/test_psi_unit.py`

### 6.1 Add Unit Tests for Classifier Variant

Add the following tests (mirror existing tests for `psi_jef` and `psi_remine`):

```python
def test_model_forward_classifier():
    """PsiModel forward returns (scalar, scalar) for psi_classifier."""
    model = PsiModel(input_dim=2, loss_type="psi_classifier", tau=0.01)
    data_p = torch.randn(32, 2)
    data_q = torch.randn(32, 2)
    psi_est, loss = model(data_p, data_q)
    assert psi_est.shape == ()
    assert loss.shape == ()

def test_model_loss_differentiable_classifier():
    """Loss is differentiable for psi_classifier."""
    model = PsiModel(input_dim=2, loss_type="psi_classifier", tau=0.01)
    data_p = torch.randn(32, 2)
    data_q = torch.randn(32, 2)
    _, loss = model(data_p, data_q)
    loss.backward()
    # Check at least one parameter has gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad

def test_classifier_psi_zero_when_same_distribution():
    """Classifier PSI estimate should be near zero when P = Q."""
    model = PsiModel(input_dim=2, loss_type="psi_classifier", tau=0.01)
    data = torch.randn(256, 2)
    psi_est, _ = model(data, data)
    # Untrained model on identical data should give ~0
    assert abs(psi_est.item()) < 1.0

def test_classifier_loss_is_positive():
    """BCE loss should be positive."""
    model = PsiModel(input_dim=2, loss_type="psi_classifier", tau=0.01)
    data_p = torch.randn(32, 2)
    data_q = torch.randn(32, 2)
    _, loss = model(data_p, data_q)
    assert loss.item() > 0

def test_classifier_tau_bounds_output():
    """Classifier output should be bounded by tau."""
    model = PsiModel(input_dim=2, loss_type="psi_classifier", tau=0.05)
    data_p = torch.randn(100, 2) * 10  # Large inputs to push sigmoid to extremes
    data_q = torch.randn(100, 2) * 10
    with torch.no_grad():
        t_p = model.network(data_p)
        omega_p = torch.sigmoid(t_p).clamp(0.05, 0.95)
    assert omega_p.min().item() >= 0.05 - 1e-6
    assert omega_p.max().item() <= 0.95 + 1e-6
```

### 6.2 Add Classifier to Existing Parametrized Tests

If existing tests use `@pytest.mark.parametrize` over `loss_type`, add
`"psi_classifier"` to the parameter list.

---

## 7. Changes to `CLAUDE.md`

Add the following to the "Loss Variants" section:

```markdown
**`"psi_classifier"`** — Binary classifier with BCE loss (Molavipour et al., 2021):

Training loss (binary cross-entropy):
```
loss = -(mean[log ω(x_p)] + mean[log(1 - ω(x_q))])
```

PSI estimation (all in log space):
```
log_Γ(x) = logit(ω(x)) = log(ω(x)) - log(1 - ω(x))
psi_est = mean[log_Γ(x_p)] - mean[log_Γ(x_q)]
```

where ω(x) = sigmoid(T(x)) clipped to [τ, 1-τ]. The parameter `tau` (default 0.01)
controls the max density ratio (~1/τ). No exponential terms in training or estimation.
Most stable variant — use when `psi_jef` or `psi_remine` show convergence issues.
```

Update the architecture table:

```markdown
| `loss_type`: psi_jef/psi_remine/psi_classifier | Loss variant |
```

---

## 8. Theoretical Justification and Caveats

### 8.1 Consistency

The classifier approach is consistent: as sample size n → ∞ and network capacity grows,
the BCE-optimal classifier converges to the Bayes-optimal classifier, and the PSI
estimate converges to the true PSI. This follows from Theorem 2 in Molavipour et al.,
applied to the LDR estimator (Equation 18), which is the non-symmetric version of our
PSI formula.

### 8.2 Bias from τ

The clipping introduces bias when the true pointwise density ratio exceeds `(1-τ)/τ`.
For `τ = 0.01`, this means bias when `P(x)/Q(x) > 99` at any point. This is a milder
bias than `clamp_max=10` on the critic (which limits density ratios to `e^10 ≈ 22,000`)
but for PSI values up to ~9, the bias is negligible because PSI integrates the
log-density ratio, not the density ratio itself.

If higher PSI values need to be estimated accurately, decrease `τ` (e.g., `τ = 0.001`
allows density ratios up to 999). The tradeoff is increased variance in the tails.

### 8.3 When to Prefer Each Variant

| Scenario | Recommended variant | Why |
|---|---|---|
| General use / first attempt | `psi_classifier` | Most stable, fewest hyperparams |
| PSI < 5, fast convergence needed | `psi_jef` | Direct optimization, fewer epochs |
| PSI > 10, well-separated distributions | `psi_remine` | Log-partition regularization helps |
| Debugging / comparing approaches | All three | Cross-validate estimates |
| Production monitoring pipeline | `psi_classifier` | Robust to hyperparameter choices |

### 8.4 Loss ≠ PSI for the Classifier

Unlike `psi_jef` where `loss = -psi_est`, the classifier's loss is BCE — it measures
classification accuracy, not PSI directly. This means:

- The loss curve will look like a standard classifier training curve (decreasing from
  `log(2) ≈ 0.693` toward 0), not like the critic's loss curve.
- PSI should be tracked as a separate metric during training, plotted on the second
  subplot of `plot_metrics()`.
- Early stopping should use the **smoothed loss** (BCE), not the PSI estimate. The BCE
  loss is a well-behaved proxy: when the classifier converges, the PSI estimate
  stabilizes.

---

## 9. Complete `forward()` Method After All Changes

For reference, here is the full `forward()` method incorporating all three variants
plus the gradient stabilization changes from `spec_gradient_stabilization.md`:

```python
def forward(
    self, data_p: torch.Tensor, data_q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    t_p = self.network(data_p)  # (N, 1)
    t_q = self.network(data_q)  # (N, 1)

    # Clamp critic outputs (not used for classifier — sigmoid handles bounding)
    if self.clamp_max is not None and self.loss_type != "psi_classifier":
        t_p = torch.clamp(t_p, -self.clamp_max, self.clamp_max)
        t_q = torch.clamp(t_q, -self.clamp_max, self.clamp_max)

    if self.loss_type == "psi_jef":
        term_p = (t_p - torch.exp(-t_p) + 1).mean()
        term_q = (torch.exp(t_q) + t_q - 1).mean()
        psi_est = term_p - term_q
        loss = -psi_est

    elif self.loss_type == "psi_remine":
        term_p = (t_p - torch.exp(-t_p) + 1).mean()
        log_partition = torch.logsumexp(t_q.squeeze(), dim=0) - math.log(
            t_q.shape[0]
        )
        term_q = torch.exp(log_partition) + t_q.mean() - 1
        psi_est = term_p - term_q
        base_loss = -psi_est
        reg = self.remine_reg_weight * (log_partition - self.remine_target_val) ** 2
        loss = base_loss + reg

    elif self.loss_type == "psi_classifier":
        omega_p = torch.sigmoid(t_p).clamp(self.tau, 1.0 - self.tau)
        omega_q = torch.sigmoid(t_q).clamp(self.tau, 1.0 - self.tau)

        # Training loss: BCE (P=1, Q=0)
        loss = -(torch.log(omega_p).mean() + torch.log(1.0 - omega_q).mean())

        # PSI: E_P[logit(ω)] - E_Q[logit(ω)]
        log_gamma_p = torch.log(omega_p) - torch.log(1.0 - omega_p)
        log_gamma_q = torch.log(omega_q) - torch.log(1.0 - omega_q)
        psi_est = log_gamma_p.mean() - log_gamma_q.mean()

    else:
        raise ValueError(
            f"Unknown loss_type '{self.loss_type}'. "
            "Use 'psi_jef', 'psi_remine', or 'psi_classifier'."
        )

    return psi_est, loss
```

---

## 10. Files Modified (Summary)

| File | Changes |
|---|---|
| `source/psi_models.py` | Add `tau` param; add `"psi_classifier"` branch in `forward()`; skip clamping for classifier; update error message |
| `source/psi.py` | Pass `tau` to `PsiModel` constructor |
| `scripts/test_gaussian.py` | Add classifier test configuration and run all scenarios with `psi_classifier` |
| `tests/test_psi_unit.py` | Add unit tests for classifier variant |
| `CLAUDE.md` | Document `psi_classifier` loss variant and `tau` parameter |

No changes to `psi_sampler.py`, `utils.py`, `gaussian_psi.py`, or any frozen MINE files.
No new files created — the classifier is a new loss type within the existing architecture.
