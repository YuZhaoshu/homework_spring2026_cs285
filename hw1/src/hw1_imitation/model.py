"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.output_dim = chunk_size * action_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.net(state)
        target = action_chunk.view(action_chunk.shape[0], -1)
        return F.mse_loss(pred, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred = self.net(state)
        return pred.view(state.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.output_dim = chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + self.output_dim + 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        target = action_chunk.view(action_chunk.shape[0], -1)
        noise = torch.randn_like(target)
        tau = torch.rand(
            target.shape[0],
            1,
            device=target.device,
            dtype=target.dtype,
        )
        x_tau = (1.0 - tau) * noise + tau * target
        model_in = torch.cat([state, x_tau, tau], dim=1)
        pred = self.net(model_in)
        target_velocity = target - noise
        return F.mse_loss(pred, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x = torch.randn(
            batch_size,
            self.output_dim,
            device=state.device,
            dtype=state.dtype,
        )
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            tau = torch.full(
                (batch_size, 1),
                step * dt,
                device=state.device,
                dtype=state.dtype,
            )
            model_in = torch.cat([state, x, tau], dim=1)
            velocity = self.net(model_in)
            x = x + velocity * dt
        return x.view(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
