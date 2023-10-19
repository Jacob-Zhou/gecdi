# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        label_smoothing = self.label_smoothing if self.training else 0.
        return F.cross_entropy(input, target,
                               weight=self.weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction,
                               label_smoothing=label_smoothing)
