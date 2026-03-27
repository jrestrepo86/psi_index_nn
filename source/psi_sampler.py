import numpy as np
import torch


class PsiSampler:
    def __init__(
        self,
        X_p: np.ndarray,
        X_q: np.ndarray,
        rng_seed: int | None = None,
    ):
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
