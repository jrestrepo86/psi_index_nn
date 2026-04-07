import math

import torch
import torch.nn as nn

from source.utils import get_activation_fn


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
        super().__init__()

        hidden_layers = [int(hl) for hl in hidden_layers]

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_layers[0]), activation_fn()]
        for i in range(len(hidden_layers) - 1):
            seq += [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(hidden_layers[-1], 1)]
        self.network = nn.Sequential(*seq)

        self.loss_type = loss_type.lower()
        self.remine_reg_weight = remine_reg_weight
        self.remine_target_val = remine_target_val
        self.clamp_max = clamp_max
        self.tau = tau

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
            # Apply sigmoid + clip to get classifier output in [tau, 1-tau]
            omega_p = torch.sigmoid(t_p).clamp(self.tau, 1.0 - self.tau)
            omega_q = torch.sigmoid(t_q).clamp(self.tau, 1.0 - self.tau)

            # Training loss: binary cross-entropy (P=1, Q=0)
            loss = -(torch.log(omega_p).mean() + torch.log(1.0 - omega_q).mean())

            # PSI estimation: E_P[logit(ω)] - E_Q[logit(ω)]
            log_gamma_p = torch.log(omega_p) - torch.log(1.0 - omega_p)
            log_gamma_q = torch.log(omega_q) - torch.log(1.0 - omega_q)
            psi_est = log_gamma_p.mean() - log_gamma_q.mean()

        else:
            raise ValueError(
                f"Unknown loss_type '{self.loss_type}'. "
                "Use 'psi_jef', 'psi_remine', or 'psi_classifier'."
            )

        return psi_est, loss
