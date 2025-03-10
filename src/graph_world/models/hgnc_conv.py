# hgcn_conv.py
# Copyright ...
# Licensed under the Apache License, Version 2.0.
# Implementation of a hyperbolic graph convolutional layer,
# loosely based on "Hyperbolic Graph Convolutional Networks" by Chami et al.

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

def mobius_add(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    # Simple implementation of Mobius addition in the Poincaré ball model.
    # Note: In practice, consider using a robust library or a more complete implementation.
    x2 = (x**2).sum(dim=-1, keepdim=True)
    y2 = (y**2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c**2 * x2 * y2 + 1e-5
    return numerator / denominator

def exp_map_zero(x: Tensor, c: float = 1.0) -> Tensor:
    # Exponential map at the origin: map from tangent space to the Poincaré ball.
    norm = x.norm(dim=-1, keepdim=True)
    return torch.tanh(torch.sqrt(c) * norm) * x / (torch.sqrt(c) * norm + 1e-5)

def log_map_zero(x: Tensor, c: float = 1.0) -> Tensor:
    # Logarithmic map at the origin: map from the Poincaré ball to tangent space.
    norm = x.norm(dim=-1, keepdim=True)
    return torch.atanh(torch.sqrt(c) * norm) * x / (torch.sqrt(c) * norm + 1e-5)

class HGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, c: float = 1.0, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.linear = nn.Linear(in_channels, out_channels)
        self.c = c
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Map input features from tangent space to hyperbolic space.
        x_hyp = exp_map_zero(x, c=self.c)
        # Linear transformation in hyperbolic space.
        x_transformed = self.linear(x_hyp)
        # Perform message passing (here using standard aggregation).
        out = self.propagate(edge_index, x=x_transformed)
        # Map the result back to the tangent space.
        out = log_map_zero(out, c=self.c)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
