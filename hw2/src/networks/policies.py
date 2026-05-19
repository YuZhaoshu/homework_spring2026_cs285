import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            # 构建一个输出维度为动作类别数（ac_dim）的网络 logits_net，去预测每个动作的 Logit（未归一化的概率对数）。
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            # 构建一个预测高斯分布均值的网络 mean_net，同时定义一个独立于状态的可学习参数 logstd（对数标准差）。
            # 将均值网络的参数和 logstd 用 itertools.chain 链在一起交给优化器。
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        # 最后，实例化一个 Adam 优化器来更新这些参数。
        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    # @torch.no_grad() 装饰器表示在这个函数内不进行梯度图跟踪
    # 这在利用策略和环境交互收集数据（Rollout）时非常重要，能极大节省内存和算力。
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_t = ptu.from_numpy(obs)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        dist = self.forward(obs_t)
        # 使用 .sample() 按照计算出的概率密度去随机采样出一个动作
        action_t = dist.sample()
        action = ptu.to_numpy(action_t)
        return action[0]

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # 离散的，返回一个类别分布类型 Categorical。
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            # 连续的，首先通过指数函数把 logstd 变成标准差 std
            # 然后利用网络输出的 mean 去构建一个多元高斯/正态分布 Normal。
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd).expand_as(mean)
            dist = distributions.Normal(mean, std)
            # 通常对多维连续动作，神经网络在每一维输出一个独立正态分布
            # 用以下代码包装一下后，它在计算 log_prob 时会把同一个动作的各个维度的概率密度相乘（即对数概率相加）
            # 作为一个整体的多维联合分布来评估该特定动作向量的可能性。
            return distributions.Independent(dist, 1)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the policy gradient actor loss
        if self.discrete:
            actions = actions.long()
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        loss = -(log_prob * advantages).mean() # (14)

        # TODO: perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
