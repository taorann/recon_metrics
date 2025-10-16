import math
import os
import random

from PIL import Image
import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from guided_diffusion import logger


def _dist_info():
    """
    Return (rank, world_size) if torch.distributed is initialized; otherwise (0, 1).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def load_data_for_reverse(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=True,
        random_crop=False,
        random_flip=False,
):
    """
    【用途】
    针对“反演/重建（reverse）”任务构建一个数据生成器（Python 生成器），
    每次产出一个 batch：(images, kwargs, paths)
    - images: 形状为 (N, C, H, W) 的 float32 张量，数值范围在 [-1, 1]。
    - kwargs: 字典，可能包含键 "y"（类别标签，int64），便于类条件模型使用。
    - paths: 原始图像文件的绝对/相对路径列表（用于保存时还原文件名/子目录）。

    【与 load_data 的区别】
    - 该函数会“保留路径信息 paths”，并返回三元组 (images, kwargs, paths)，
      以方便反演/重建后将“重建图”和“DIRE图”映射回原输入的文件结构。
    - 同时它会在分布式(DDP)下按 rank/world_size 进行索引切片（shard），各进程处理不同切片。

    参数说明：
    - data_dir: 数据集根目录，内部可以包含子文件夹（如类别名），会被递归遍历。
    - batch_size: 批大小。
    - image_size: 统一到的目标分辨率（会中心裁剪/缩放），得到正方形 H=W=image_size。
    - class_cond: 是否需要返回类别标签 "y"。若为 True，会以“文件名下划线前缀”作为类别名。
    - deterministic: True 表示使用固定顺序（不打乱、不丢弃尾部不足一个 batch 的样本）。
    - random_crop: 是否随机裁剪（数据增强）。若 False，则中心裁剪。
    - random_flip: 是否随机水平翻转（数据增强）。
    """
    if not data_dir:
        # 未提供数据目录，直接报错，避免静默失败
        raise ValueError("unspecified data directory")

    # ===== 1) 收集全部图像文件路径（各 rank 独立扫描，结果一致即可）=====
    all_files = _list_image_files_recursively(data_dir)
    rank, world = _dist_info()

    # ===== 2)（可选）构建类别标签 =====
    classes = None
    if class_cond:
        # 约定：类别名为“文件名中下划线前的前缀”
        # 例如：cat_0001.png -> 类别 "cat"
        class_names = [bf.basename(path).split("_")[0] for path in all_files]  # 标签_文件名
        # 将去重后的类别名按字典序排序并映射到 [0..num_classes-1] 的整数标签
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # 把每个样本的类别名转成整数标签列表（与 all_files 对齐）
        classes = [sorted_classes[x] for x in class_names]

    # ===== 3) 构建数据集对象（带 MPI 分片）=====
    # ImageDataset_for_reverse 的特点：
    # - 返回 (tensor_image, out_dict, path) 三元组，其中 out_dict 可能含 "y"
    # - 会根据 shard/num_shards 对 all_files 做切片，以便多进程并行不重叠
    dataset = ImageDataset_for_reverse(
        image_size,
        all_files,
        classes=classes,
        shard=rank,            # 当前进程的分片索引
        num_shards=world,      # 分片总数（=进程数）
        random_crop=random_crop,
        random_flip=random_flip,
    )

    # 记录数据集在“全局”下的长度（单分片长度 * 进程数）
    logger.log("dataset length: {}".format(dataset.__len__() * (world if world else 1)))

    # ===== 4) 构建 DataLoader（是否打乱、是否丢弃不足一个 batch 的尾部，取决于 deterministic）=====
    if deterministic:
        # 复现/评测场景常用：不打乱；不丢弃尾部（drop_last=False）
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
    else:
        # 训练/随机采样场景：打乱；为了对齐 batch 大小可设置 drop_last=True
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )

    # ===== 5) 返回一个“无限”生成器 =====
    # 这里使用 while True + yield from 的写法，不断从 DataLoader 里产出 batch。
    # 外部调用方（如 compute_dire.py）会按自身的 stopping 条件（num_samples）控制循环退出。
    while True:
        yield from loader


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    rank, world = _dist_info()
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=rank,
        num_shards=world,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset_for_reverse(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        arr = torch.from_numpy(np.transpose(arr, [2, 0, 1]))
        return arr, out_dict, path


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
