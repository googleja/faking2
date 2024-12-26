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
from torchvision import transforms as T


"""这是当初为了测试有img buf后make batch以及bath to data的代码，
目前已完全移植到loraDino/net_ft/train.py中"""


def img_to_torch(img_path, device="cuda"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_tensor = T.ToTensor()
    return to_tensor(img).to(device)


def extract_mask(ann_path, target_rgb):
    image = cv2.imread(ann_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(image_rgb == target_rgb, axis=-1)

    # mask_for_show = mask.astype(np.uint8) * 255
    # print(ann_path)
    # cv2.imwrite("/home/jack/wvn/SurgicalDINO/data1219/annotation/mask.png", mask_for_show)
    # cv2.imshow("Mask", mask_for_show)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    return mask


def make_batch(batch_size=4):
    img_dir = "/home/jack/wvn/SurgicalDINO/data1219/image/"
    ann_dir = "/home/jack/wvn/SurgicalDINO/data1219/annotation/"

    img_list = random.sample(os.listdir(img_dir), batch_size)

    img_buf = []
    ann_buf = []
    for img in img_list:
        img_buf.append(img_to_torch(img_dir + img))
        mask = extract_mask(ann_dir + img, [255, 229, 204])
        mask_resized = cv2.resize(mask.astype(np.uint8), (28, 28), interpolation=cv2.INTER_NEAREST)
        ann_buf.append(mask_resized)

    return torch.stack(img_buf), ann_buf


def batch_to_data(img_batch, masks, batch_size=4):
    # step2: 根据img和mask生成正负样本，并判断是否达到最小训练数据量"""
    feats = torch.randn(batch_size, 90, 28, 28)

    positive_data = []
    negative_data = []

    for i, mask in enumerate(masks):
        positive_indices = np.where(mask == 1)  # 掩膜为1的位置
        negative_indices = np.where(mask == 0)

        # 转置后形状为[num_positive, 90]和[num_negative, 90]
        # a = feats[i][:, positive_indices[0], positive_indices[1]]
        # b = feats[i][:, negative_indices[0], negative_indices[1]]
        positive = feats[i][:, positive_indices[0], positive_indices[1]]
        negative = feats[i][:, negative_indices[0], negative_indices[1]]
        positive_data.append(positive.T)
        negative_data.append(negative.T)
        print(f"Positive data size: {len(positive.T)}, Negative data size: {len(negative.T)}")

        # num_samples = len(self.positive_data)
        # if num_samples > self._min_samples_for_training:
        # TODO 要判断是否达到最小训练数据量，不够的话再重新选择

    positive_data = torch.cat(positive_data, dim=0)
    negative_data = torch.cat(negative_data, dim=0)

    if len(positive_data) % 2 != 0:
        positive_data = positive_data[:-1]  # 去除一个样本以确保长度为偶数

    random.shuffle(positive_data)
    half_size = len(positive_data) // 2
    query = positive_data[:half_size]
    positive_key = positive_data[half_size:]

    return query, positive_key, negative_data


def test_datagen():
    # step1: 从img_buffer中make batch
    img_batch, masks = make_batch()
    query, positive_key, negative_data = batch_to_data(img_batch, masks)
    return 0


if __name__ == "__main__":
    test_datagen()
