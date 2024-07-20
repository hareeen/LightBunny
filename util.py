from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(
        m,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 2,
        delta: float = 0,
        path: str | Path = "ephemeral_best_weights.pth",
    ):
        self.min_loss = np.inf
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0

    def evaluate(self, model: nn.Module, loss: float):
        if loss + self.delta > self.min_loss:
            self.counter += 1
        else:
            self.counter = 0

        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(model.state_dict(), self.path)

    def should_stop(self):
        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        model.load_state_dict(torch.load(self.path))
        return model
