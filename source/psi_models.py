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

    def forward(
        self, data_p: torch.Tensor, data_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_p = self.network(data_p)  # (N, 1)
        t_q = self.network(data_q)  # (N, 1)

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

        else:
            raise ValueError(
                f"Unknown loss_type '{self.loss_type}'. Use 'psi_jef' or 'psi_remine'."
            )

        return psi_est, loss
