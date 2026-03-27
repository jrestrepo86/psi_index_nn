"""
PSI: Population Stability Index neural estimation
"""

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import schedulefree
import torch
import torch.optim as optim
from plotly.subplots import make_subplots

from source.utils import (EarlyStopping, ExpMovingAverageSmooth, to_col_vector,
                          train_test_split)

from .psi_models import PsiModel
from .psi_sampler import PsiSampler


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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x_p = to_col_vector(X_p)
        self.x_q = to_col_vector(X_q)

        input_dim = self.x_p.shape[1]

        self.model = PsiModel(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            afn=afn,
            loss_type=loss_type,
            remine_reg_weight=remine_reg_weight,
            remine_target_val=remine_target_val,
        ).to(self.device)

        self.loss_type = loss_type
        self.metrics = None
        self.trained = False
        self.log_metrics = False
        self.best_state = None

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
        self.log_metrics = log_metrics

        # Optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=1000,
        )

        # self.optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr=lr,
        #     weight_decay=weight_decay,
        # )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=stop_patience,
            delta=stop_min_delta,
            warmup_steps=stop_warmup_steps,
        )

        # Exponential smooth
        smooth = ExpMovingAverageSmooth()

        # Independent train/test splits for P and Q
        train_idx_p, test_idx_p = train_test_split(
            self.x_p.shape[0], test_size, contigous=contiguous_split
        )
        train_idx_q, test_idx_q = train_test_split(
            self.x_q.shape[0], test_size, contigous=contiguous_split
        )

        training_sampler = PsiSampler(
            self.x_p[train_idx_p],
            self.x_q[train_idx_q],
        )
        testing_sampler = PsiSampler(
            self.x_p[test_idx_p],
            self.x_q[test_idx_q],
        )

        best_state, best_loss = None, float("inf")
        metrics = []

        for epoch in range(num_epochs):
            # Training
            batch_p, batch_q = training_sampler.sample(batch_size)
            batch_p = batch_p.to(self.device)
            batch_q = batch_q.to(self.device)
            self.model.train()
            self.optimizer.train()
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                _, loss = self.model(batch_p, batch_q)
                loss.backward()
                self.optimizer.step()

            # Evaluation
            batch_p, batch_q = testing_sampler.sample(batch_size)
            batch_p = batch_p.to(self.device)
            batch_q = batch_q.to(self.device)
            self.model.eval()
            self.optimizer.eval()
            with torch.no_grad():
                psi_est, loss = self.model(batch_p, batch_q)
                smoothed_loss = smooth(loss.item())

                if self.log_metrics:
                    metrics.append(
                        {
                            "epoch": epoch,
                            "loss": loss.item(),
                            "smoothed_loss": smoothed_loss,
                            "psi": psi_est.item(),
                        }
                    )

            # if smoothed_loss < best_loss:
            #     best_loss = smoothed_loss
            #     best_state = copy.deepcopy(self.model.state_dict())

            early_stopping(smoothed_loss)
            if early_stopping.early_stop:
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

        self.trained = True

        if log_metrics:
            self.metrics = pd.DataFrame(metrics)

    def get_psi(self) -> float:
        if not self.trained:
            raise ValueError("Did you call .train()?")

        sampler = PsiSampler(self.x_p, self.x_q)
        batch_p, batch_q = sampler.sample(self.x_p.shape[0])
        batch_p = batch_p.to(self.device)
        batch_q = batch_q.to(self.device)
        self.model.eval()
        self.optimizer.eval()
        with torch.no_grad():
            psi_est, _ = self.model(batch_p, batch_q)
        return max(0.0, psi_est.item())

    def plot_metrics(self, text: str = "", show: bool = True) -> go.Figure:
        if self.metrics is None:
            raise ValueError("No metrics to plot. Did you call .train()?")
        if self.log_metrics is False:
            print("No psi metrics recorded. Set log_metrics = True and call .train()?")

        epochs = self.metrics["epoch"].values
        loss = self.metrics["loss"].values
        smoothed_loss = self.metrics["smoothed_loss"].values
        if self.log_metrics:
            psi = self.metrics["psi"].values
        else:
            psi = np.zeros(len(epochs))

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Epochs vs Loss", "Epochs vs PSI Estimate"),
        )

        fig.add_trace(
            go.Scatter(x=epochs, y=loss, mode="lines+markers", name="Loss"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=smoothed_loss,
                mode="lines+markers",
                name="Smoothed Loss",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs, y=psi, mode="lines+markers", name=f"{self.loss_type.upper()}"
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title_text="PSI | Training Metrics" + " " + text,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="PSI", row=2, col=1)

        if show:
            fig.show()
        return fig
