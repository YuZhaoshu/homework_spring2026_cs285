from collections import OrderedDict
import numpy as np
import copy
from networks.policies import MLPPolicy
import gym
import cv2
from infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]: # 定义函数并加类型注解，表示返回值是一个 dict，键为字符串，值为 np.ndarray。
    """Sample a rollout in the environment from a policy."""
    # 调用 Gym 的重置接口，返回初始观测；新版本 Gym 会返回 (obs, info)。
    reset_out = env.reset()
    # 兼容新旧 Gym 的写法，若是 tuple 就取第 0 项作为观测。
    ob = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    # 创建多个列表用来缓存一条轨迹内每个时间步的数据。
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    # 计数器，用来限制轨迹最大长度。
    steps = 0
    while True: # 无限循环，直到满足终止条件。
        # render an image
        if render:
            # 如果是 MuJoCo 环境（有 sim），调用 env.sim.render；否则用 env.render。
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render()
                if img is None:
                    try:
                        img = env.render(mode="rgb_array")
                    except TypeError:
                        img = env.render(mode="single_rgb_array")
            if isinstance(img, (list, tuple)):
                if len(img) == 0:
                    img = None
                else:
                    img = img[-1]
            if img is not None:
                img = np.asarray(img)
                if (
                    img.dtype.kind not in ("u", "i", "f")
                    or img.ndim < 2
                    or img.shape[0] <= 0
                    or img.shape[1] <= 0
                ):
                    img = None
            if img is not None:
                img = np.ascontiguousarray(img)
                # 把渲染图缩成 250x250 方便存储与日志。
                try:
                    image_obs.append(
                        cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
                    )
                except cv2.error:
                    pass

        # TODO use the most recent ob to decide what to do
        # 用当前策略根据观测采样动作。这里 policy 是 MLPPolicy 或其子类。
        ac = policy.get_action(ob)

        # TODO: take that action and get reward and next ob
        # 执行一步交互。
        step_out = env.step(ac)
        if len(step_out) == 5:
            # 新 Gym 返回五元组 (obs, reward, terminated, truncated, info)
            next_ob, rew, terminated, truncated, info = step_out
            # 把两种终止原因合成一个布尔值。
            done = terminated or truncated
        else:
            # 旧 Gym 返回四元组 (obs, reward, done, info)。
            next_ob, rew, done, info = step_out

        # TODO rollout can end due to done, or due to max_length
        # 每步增加计数。
        steps += 1
        # 当环境结束或达到最大步数就终止。
        rollout_done = done or steps >= max_length

        # record result of taking that action
        # 把当前时刻的观测、动作、奖励、下一观测、终止标记保存到轨迹缓存列表中。
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        # 推进到下一时刻。
        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        # 一旦终止条件满足，退出采样。
        if rollout_done:
            break

    # 把各列表转成 np.ndarray，并固定 dtype：观测和动作转成 float32，奖励转成 float32，图像转成 uint8。
    # terminal 里存的是每一步是否为“轨迹结束步”，用于后续优势估计与 GAE 计算。
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
