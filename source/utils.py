import numpy as np
import torch.nn as nn
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split as sk_train_test_split

rng = np.random.default_rng()


def to_col_vector(x):
    """
    Convert a 1D array to a column vector.
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def scale(x: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def get_activation_fn(afn: str = "relu"):
    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        "softmax": nn.Softmax,
        "gelu": nn.GELU,
    }

    if afn not in activation_functions:
        raise ValueError(
            f"'{afn}' is not included in activation_functions. Use below one \n {activation_functions.keys()}"
        )

    return activation_functions[afn]


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        warmup_steps: int = 100,
    ):
        self.patience = patience
        self.delta = abs(delta)
        self.warmup_steps = warmup_steps
        self.epoch = 0  # track total epochs
        self.no_improve_counter = 0  # track patience after warmup
        self.min_loss = np.inf
        self.early_stop = False

    def __call__(self, loss: float):
        # Always increment epoch count
        self.epoch += 1

        # If still in warmup phase, just update min_loss and return
        if self.epoch <= self.warmup_steps:
            if loss < self.min_loss:
                self.min_loss = loss
            return

        # After warmup, check for improvement
        if loss < self.min_loss - self.delta:
            self.min_loss = loss
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1
            if self.no_improve_counter >= self.patience:
                self.early_stop = True


class ExpMovingAverageSmooth:
    def __init__(self, alpha=0.01):
        self.ema = None
        self.alpha = alpha

    def __call__(self, loss):
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.alpha * loss + (1.0 - self.alpha) * self.ema
        return self.ema


def train_test_split(
    data_size: int,
    test_size: float,
    contigous: bool = False,
    rng_seed: int | None = None,
):
    """
    Split indices into train/test sets. Optionally enforce contiguous test block.

    Parameters
    ----------
    data_size : int
        Total number of samples.
    test_size : float
        Fraction of data to use for test (0 < test_size < 1).
    contiguous : bool, default=False
        If True, test set will be a contiguous block of indices.
    rng_seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        Indices for training.
    test_idx : np.ndarray
        Indices for testing.
    """
    rng = np.random.default_rng(rng_seed)

    if contigous:
        n_test = int(test_size * data_size)
        max_start = data_size - n_test
        test_start_idx = rng.integers(0, max_start + 1)  # scalar
        test_end_idx = test_start_idx + n_test
        test_idx = np.arange(test_start_idx, test_end_idx)
        train_idx = np.setdiff1d(np.arange(data_size), test_idx)
    else:
        train_idx, test_idx = sk_train_test_split(
            list(range(data_size)), test_size=test_size, shuffle=True
        )
    return train_idx, test_idx
