import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque


def extract_mask(annotation_path, img_path=None, target_range=((250, 20, 10), (255, 30, 20))):
    """
    根据指定颜色范围从注释图片中提取类别掩码。

    Args:
        annotation_path (str): 输入注释图片路径。
        target_color (tuple): 目标颜色 (R, G, B)。
    """
    annotation = Image.open(annotation_path).convert("RGB")
    annotation_array = np.array(annotation)

    mask = np.all((annotation_array >= target_range[0]) & (annotation_array <= target_range[1]), axis=-1)

    # if img_path:
    #     test_image = Image.open(img_path).convert("RGB")
    #     test_image_array = np.array(test_image)
    #     overlay_image = cv2.addWeighted(test_image_array, 0.7, np.stack([mask]*3, axis=-1).astype(np.uint8) * 255, 0.3, 0)
    #
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(overlay_image)
    #     plt.axis("off")
    #     plt.title("Image with Mask Overlay")
    #     plt.savefig("/home/jack/wvn/SurgicalDINO/data1219/image/overlay_image.png")
    #     plt.show()

    return mask


def generate_data(features, mask):
    positive_indices = np.where(mask == 1)  # 掩膜为1的位置
    negative_indices = np.where(mask == 0)  # 掩膜为0的位置

    positive_samples = features[:, positive_indices[0], positive_indices[1]]
    negative_samples = features[:, negative_indices[0], negative_indices[1]]
    # 将正负样本对返回，形状为[num_positive, 90]和[num_negative, 90]
    return positive_samples.T, negative_samples.T


class DataBuffer:
    def __init__(self, max_size):
        self.positive_buffer = deque(maxlen=max_size)
        self.negative_buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add_positive(self, data):
        """添加query或positive数据到buffer"""
        self.positive_buffer.extend(data)

    def add_negative(self, data):
        """添加negative数据到buffer"""
        self.negative_buffer.extend(data)

    def sample_query_and_positive(self, batch_size):
        """随机采样query和positive样本"""
        if len(self.positive_buffer) < batch_size:
            raise ValueError("Not enough positive data in buffer to sample")

        query = torch.stack(random.sample(self.positive_buffer, batch_size))
        positive_key = torch.stack(random.sample(self.positive_buffer, batch_size))

        return query, positive_key

    def sample_negative(self, size):
        """根据infoNCE，负样本要多采一些比较合适"""
        if len(self.negative_buffer) < size:
            raise ValueError("Not enough negative data in buffer to sample")

        negative_key = torch.stack(random.sample(self.negative_buffer, size))

        return negative_key

    def size(self):
        return len(self.positive_buffer)

    def is_ready(self, min_positive_size=64, min_negative_size=128):
        """检查是否达到最小训练数据量"""
        return len(self.positive_buffer) >= min_positive_size and len(self.negative_buffer) >= min_negative_size

    def __len__(self):
        return len(self.positive_buffer)

    def generate_data(self, feats, mask):
        positive_indices = np.where(mask == 1)  # 掩膜为1的位置
        negative_indices = np.where(mask == 0)  # 掩膜为0的位置
        # 转置后形状为[num_positive, 90]和[num_negative, 90]
        self.add_positive(feats[:, positive_indices[0], positive_indices[1]].T)
        self.add_negative(feats[:, negative_indices[0], negative_indices[1]].T)
        print(f"Positive buffer size: {len(self.positive_buffer)}, Negative buffer size: {len(self.negative_buffer)}")

# class ContrastiveDataset(Dataset):
#     """设置negatives的数量与positives一样"""
#     def __init__(self, root_dir, transform=None):
#         """
#         带文件夹划分的对比学习数据集
#         :param root_dir: 数据集的根目录
#         :param transform: 数据增强方法
#         :param num_negatives: 每个样本的负样本数量
#         """
#         self.query_dir = os.path.join(root_dir, "query")
#         self.positive_dir = os.path.join(root_dir, "positive")
#         self.negative_dir = os.path.join(root_dir, "negative")
#         self.query_files = sorted([f for f in os.listdir(self.query_dir) if f.endswith('.pt')])
#         self.positive_files = sorted([f for f in os.listdir(self.positive_dir) if f.endswith('.pt')])
#         self.negative_files = sorted([f for f in os.listdir(self.negative_dir) if f.endswith('.pt')])
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.query_files)
#
#     def __getitem__(self, idx):
#         """引入时就是tensor了"""
#         query_path = os.path.join(self.query_dir, self.query_files[idx])
#         query_img = torch.load(query_path)
#
#         positive_path = os.path.join(self.positive_dir, self.positive_files[idx])
#         positive_img = torch.load(positive_path)
#
#         negative_path = os.path.join(self.negative_dir, self.negative_files[idx])
#         negative_img = torch.load(negative_path)
#
#         if self.transform:
#             query_img = self.transform(query_img)
#             positive_img = self.transform(positive_img)
#             negative_img = self.transform(negative_img)
#
#         return query_img, positive_img, negative_img


if __name__ == "__main__":

    buffer = DataBuffer(max_size=200)
    test_img = "/home/jack/wvn/SurgicalDINO/data1219/image/trail-15_00361.png"
    test_ann = "/home/jack/wvn/SurgicalDINO/data1219/annotation/trail-15_00361.png"
    mask = extract_mask(test_ann, test_img, target_range=((240, 220, 195), (255, 235, 215)))
    mask_resized = cv2.resize(mask.astype(np.uint8), (28, 28), interpolation=cv2.INTER_NEAREST)

    # plt.figure(figsize=(4, 4))
    # plt.imshow(mask_resized, cmap='gray')
    # plt.axis("off")
    # plt.title("Resized Mask")
    # plt.show()

    features = np.random.randn(90, 28, 28)  # 模拟Stego提取的特征

    positive_samples, negative_samples = generate_data(features, mask_resized)

    buffer.add_positive(positive_samples)
    buffer.add_negative(negative_samples)

    if buffer.is_ready(50, 100):
        query, positive_keys = buffer.sample_query_and_positive(25)
        negative_keys = buffer.sample_negative(50)

