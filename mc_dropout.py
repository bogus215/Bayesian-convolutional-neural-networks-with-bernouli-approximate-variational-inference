from torch import nn


class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5, force_dropout: bool = False):
        super().__init__()
        self.force_dropout = force_dropout
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training or self.force_dropout)
