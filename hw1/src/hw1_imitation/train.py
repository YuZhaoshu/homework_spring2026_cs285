"""Train and evaluate a Push-T imitation policy."""

# 启用 Python 3.7+ 的延迟类型求值。这意味着你可以在类定义完成前互相引用类型（防止循环引用报错）。
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
# tyro: 一个非常现代但简洁的命令行解析库，能直接把 @dataclass 转成终端的长指令传参。
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"

# 使用了 Python 的数据类装饰器 @dataclass。所有配置属性都自带了类型提示（type hints，比如 int、tuple 等）。
@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "flow" # mse
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    # 调用 tyro.cli 它可以自动读取我们在终端里敲的命令
    # 例如 python train.py --lr 1e-3，它会自动覆盖 TrainConfig 里的默认值，返回一个配置实例。
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    # DataLoader 则是 PyTorch 的批处理引擎。
    # batch_size=128 会并行把 128 个数据像搭积木一样拼成一个大张量（Tensor）
    # shuffle=True 意味着每个 epoch 取数据的顺序会完全打乱（防止模型死记硬背）
    # drop_last=True 是为了丢弃最后不够 128 个的零碎数据，避免张量形状突变引起代码报错
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size, # DataLoader 会在后台并行调用 128 次 dataset[i]，输出的形状就会变成 (128, 状态维度)
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device) # 把模型定义里所有的神经元权重（Weight）和偏置（Bias）全都从计算机主内存（RAM）推送到显卡显存（VRAM）中去。

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    # Adam 是深度学习中最主流的自适应学习率优化算法。
    # model.parameters() 是把模型里所有待训练的参数引用打包交给它托管。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    def train_step(state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(state, action_chunk)
        loss.backward()
        optimizer.step()
        return loss

    if hasattr(torch, "compile"):
        try:
            train_step = torch.compile(train_step)
        except Exception:
            pass

    global_step = 0
    model.train()
    header_logged = False

    for epoch in range(config.num_epochs):
        for state, action_chunk in loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            loss = train_step(state, action_chunk)
            global_step += 1

            if global_step % config.log_interval == 0:
                log_data = {
                    "train/loss": float(loss.item()),
                    "train/epoch": epoch,
                }
                if not header_logged:
                    log_data["eval/mean_reward"] = float("nan")
                    header_logged = True
                logger.log(log_data, step=global_step)

            if global_step % config.eval_interval == 0:
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger,
                )
                model.train()

    if global_step % config.eval_interval != 0:
        evaluate_policy(
            model=model,
            normalizer=normalizer,
            device=device,
            chunk_size=config.chunk_size,
            video_size=config.video_size,
            num_video_episodes=config.num_video_episodes,
            flow_num_steps=config.flow_num_steps,
            step=global_step,
            logger=logger,
        )

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
