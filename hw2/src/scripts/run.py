import argparse
import os
from datetime import datetime
import time

import gym
import numpy as np
import torch
import tqdm

from agents.pg_agent import PGAgent
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

MAX_NVIDEO = 2


def run_training_loop(logger, args):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 根据参数决定是否使用 GPU，设置全局设备。
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    render_mode = "rgb_array" if args.video_log_freq != -1 else None
    env = gym.make(args.env_name, render_mode=render_mode)
    # 判断动作空间是否离散。
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # 如果传了 --ep_len 就用它，否则用环境默认最大步数。or 是短路逻辑，前者为真就取前者。
    max_ep_len = args.ep_len or env.spec.max_episode_steps

    # 观测维度（假设一维向量）。
    ob_dim = env.observation_space.shape[0]
    # 若动作离散，维度是动作数 n；否则是连续动作向量长度。
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"): # 判断是否是 MuJoCo 类环境（有 model）。
        fps = 1 / env.dt
    # 用仿真时间步或环境元数据得到帧率，用于保存视频。
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    # 传入的参数包括折扣因子、网络大小、是否用 baseline、是否 reward-to-go、是否归一化优势、是否使用 GAE 等。
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter): # 循环 n_iter 次，每次迭代代表一次“采样→更新”。
        print(f"\n********** Iteration {itr} ************") # f-string 语法，打印迭代编号。
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        ) # 从当前策略采样轨迹，直到达到 args.batch_size 的步数。返回轨迹列表和本轮采样步数。
        total_envsteps += envsteps_this_batch # 累计环境交互步数，用于横轴统计。

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        # 字典推导式，把“list of dicts”转成“dict of lists”，每个 key 映射到一组轨迹对应字段（如 observation）。
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}
        # trajs = [
        #     {"x": 1, "y": 2},
        #     {"x": 3, "y": 4},
        #     {"x": 5, "y": 6}
        # ]
        # 变成：
        # {
        #     "x": [1, 3, 5],
        #     "y": [2, 4, 6]
        # }

        # TODO: train the agent using the sampled trajectories and the agent's update function
        # 调用 PGAgent 更新策略/基线。冒号是类型提示（Python 类型注解），表示 train_info 期望是 dict。
        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0: # 按指定频率做评估。
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            ) # 再采样一批 eval_trajs，用于评估性能。

            # 计算统计指标（平均回报、方差等）。
            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            # 把训练阶段的 loss 指标合并到日志。
            logs.update(train_info)
            # 记录环境步数。
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
            # 写入 CSV 和 wandb。
            logger.log(logs, itr)
            print("Done logging...\n\n", flush=True)

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0: # 视频日志周期。
            print("\nCollecting video rollouts...")
            # 采样少量轨迹并渲染图像。
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            # 保存视频。
            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    dump_log(agent, logger, args, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = vars(args)
    setup_wandb(project='cs285_hw2', name=exp_name, config=config)
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'log.csv'))

    run_training_loop(logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
