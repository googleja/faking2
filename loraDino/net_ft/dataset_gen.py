import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
import random
import torch

"""
数据集保存的时候，就保存成了torch.tensor，
同时也同步保存一下其原始图片以供查看就好"""

class ContrastiveDataset(Dataset):
    """设置negatives的数量与positives一样"""
    def __init__(self, root_dir, transform=None):
        """
        带文件夹划分的对比学习数据集
        :param root_dir: 数据集的根目录
        :param transform: 数据增强方法
        :param num_negatives: 每个样本的负样本数量
        """
        self.query_dir = os.path.join(root_dir, "query")
        self.positive_dir = os.path.join(root_dir, "positive")
        self.negative_dir = os.path.join(root_dir, "negative")
        self.query_files = sorted([f for f in os.listdir(self.query_dir) if f.endswith('.pt')])
        self.positive_files = sorted([f for f in os.listdir(self.positive_dir) if f.endswith('.pt')])
        self.negative_files = sorted([f for f in os.listdir(self.negative_dir) if f.endswith('.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.query_files)

    def __getitem__(self, idx):
        """引入时就是tensor了"""
        query_path = os.path.join(self.query_dir, self.query_files[idx])
        query_img = torch.load(query_path)

        positive_path = os.path.join(self.positive_dir, self.positive_files[idx])
        positive_img = torch.load(positive_path)

        negative_path = os.path.join(self.negative_dir, self.negative_files[idx])
        negative_img = torch.load(negative_path)

        if self.transform:
            query_img = self.transform(query_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return query_img, positive_img, negative_img
