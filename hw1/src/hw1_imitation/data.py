"""Dataset utilities for Push-T."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


# 使用了 dataclass(frozen=True) 装饰器。
# 这会自动为类生成 __init__ 等方法，并且标明这个类的实例是**不可变（frozen）**的。
# 一旦初始化，均值和标准差就不能被修改了。
@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    # 语法上是一个静态方法（@staticmethod），它确保标准差不要太小（比如全0序列导致标准差是0，后续除法会报错）。
    # 遇到小于10^-6的值会被强行替作10^-6，保证数值稳定性。 
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    # 类方法（@classmethod），我们可以直接通过 Normalizer.from_data(states, actions) 来生成一个实例。
    # 它内部使用 np.mean 和 np.std 对 0 轴（即按特征维度）求均值和标准差。
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    # normalize_* / denormalize_*：用于将环境拿到的原始数据转为网络输入，或者把网络的输出转回环境实际能用的原始动作数值。
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True) # 确保数据目录存在，如果不存在就创建它。
    # dataset_dir / ZARR_RELATIVE_PATH 使用了运算符重载。
    # 在 pathlib.Path 中，除号 / 被重载为路径拼接符，取代了传统且容易拼接错的 os.path.join()。
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path # 如果数据已经存在，就直接返回路径，不需要重复下载和解压。

    # 再次使用 / 拼接出压缩包的保存路径。
    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        # urllib.request.urlretrieve(url, filename) 是 Python 内置库下载文件的标准方法。
        urllib.request.urlretrieve(PUSHT_URL, zip_path) # 从全局定义的 PUSHT_URL 下载文件并保存到本地的 zip_path 处。

    # with ... as ...: 是 Python 的上下文管理器语法。
    # 它能确保代码块执行完毕（无论是正常结束还是发生异常）后，自动调用 .close() 方法释放资源（这里是关闭文件句柄）。
    # "r" 表示以只读模式打开 zip 文件。
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # 打开刚刚下载好的 pusht.zip 文件，调用 .extractall(dataset_dir) 将压缩包内的所有内容一次性全解压到 dataset_dir 目录下。
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # zarr.open(..., mode="r") 以只读模式打开本地 Zarr 格式数据库（一种能高效支持大规模张量分块存写的格式）。
    root = zarr.open(zarr_path, mode="r")
    
    # 语法：root["data"]["state"] 像查字典一样逐层深入读取。
    # [:] 用切片语法一次性把 zarr 格式的整个数组全读出来。
    # np.asarray(..., dtype=np.float32) 强制转换为 NumPy 数组并指定用单精度浮点（32位），以节省内存并匹配 PyTorch 的默认浮点型。
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    
    # meta 下的 episode_ends 存着长长的一维数组，标明前一局在哪结束，用整数 (int64) 类型。
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    
    # 返回类型提示里说的 3 元组 (states, actions, episode_ends)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    # episode_ends 记录了每一局结束时的索引。
    # episode_ends[:-1] 是数组切片语法，表示取除了最后一个元素之外的所有元素。
    # ([0], ...) 将 0 作为整个数据集第一局的起点。两者拼接 (np.concatenate)，就得到了每一局的起始索引数组 starts。
    starts = np.concatenate(([0], episode_ends[:-1]))
    
    # 类型注解 (list[int]) 初始化一个空列表。相比于直接用 numpy 数组 append，
    # Python 原生列表由于不涉及内存的连续重分配，在频繁追加元素时效率更高。
    indices: list[int] = []
    
    # zip 并行遍历 starts 和 episode_ends 两个数组。
    # strict=True 是 Python 3.10 引入的新语法，它强制要求传入的所有可迭代对象长度必须完全一致，
    # 如果不一致就会抛出 ValueError，这是一种很好的防御性编程习惯。
    for start, end in zip(starts, episode_ends, strict=True):
        # last_start 计算了在当前这局游戏（episode）内，能够提取完整 chunk_size 动作序列的最后一个合法起点。
        # 因为我们后续要往后切片 chunk_size 的长度，如果起点太靠后，超出了 end，取到的动作不仅越界，还可能混入下一局的数据。
        last_start = end - chunk_size
        
        # 如果这局游戏总长度甚至都小于 chunk_size，那么 last_start 就会算出比 start 还小的值，
        # 此时遇到这种情况，直接 continue 跳过此局（即这局数据废弃，不产生任何有效起点）。
        if last_start < start:
            continue
            
        # range(start, last_start + 1) 生成从 start 到 last_start 的整数序列（因为 range 是左闭右开，所以需要 +1）。
        # .extend() 方法将这个迭代器产生的所有整数一次性全部追加到 indices 列表中，比在一个 for 循环里挨个 .append() 快得多。
        indices.extend(range(start, last_start + 1))
        
    # 循环结束后，将 Python 列表转回底层的 Numpy 数组，并将类型锁定为 64 位整型（专门用于做索引）。
    return np.asarray(indices, dtype=np.int64)


# 继承自 torch.utils.data.Dataset。
# 在 PyTorch 中，任何自定义的数据集都必须继承 Dataset，
# 并且必须实现 __len__ 和 __getitem__ 两个魔术方法（magic methods）。
class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        # Normalizer | None 是 Python 3.10+ 的联合类型提示语法，等价于 Optional[Normalizer]，表示该参数可以是 Normalizer，也可以是 None。
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        # 在初始化时，预先计算所有合法的滑动窗口起点下标，避免在每次加载数据时重复计算。
        self.indices = build_valid_indices(episode_ends, chunk_size)

    # __len__ 魔术方法，使得你可以通过 len(dataset) 直接获取数据集的总样本数大小。
    # PyTorch DataLoader 非常依赖这个方法来计算能划分出多少个 Batch 去做迭代。
    def __len__(self) -> int:
        return len(self.indices)

    # __getitem__ 魔术方法，它让你能通过下角标语法像访问列表一样提取单个数据：dataset[idx]
    # DataLoader 在组装 Batch 时，默认就是在后台开多线程（Worker），并行调用这个方法成百上千次。
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 从预先算好的合法切片池中，取出第 idx 个合法的全局起始时间戳 t。
        t = int(self.indices[idx])
        
        # 简单索引：提取对应那一帧的单一状态。
        state = self.states[t]
        
        # 切片语法：动作提取的是一段长度为 chunk_size 的未来连续序列。
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        # torch.from_numpy()：原生桥接函数，将 NumPy 数据共享内存转为底层 PyTorch Tensor（速度极快）。
        # .float()：强制转为单精度（torch.float32），这是在送入模型前必须的对齐操作。
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
