import os
import random

import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from loraDino.net_ft.dataset_gen import ContrastiveDataset
from torchvision import transforms as T

from loraDino.stego.stego import Stego
from loraDino.net_ft.infoNCE import InfoNCE
import cv2
import numpy as np

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)


# logic:
# 1、有一个img_buffer，用来存放1-2张图片或历史图片，就类似learning node的节点
# 2、train就是从img buffer中选取img及对应mask
# 3、根据mask生成正负样本，然后一次性feed进模型训练
# 因为如果是存储正负样本，那训练的输入本身就带着梯度了，这不好处理


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


class DINOWithLoRA(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super(DINOWithLoRA, self).__init__()
        self._model = Stego.load_from_checkpoint(model_path, n_image_clusters=20, strict=False)

        # freeze stego额外层的参数
        for param in self._model.segmentation_head.parameters():
            param.requires_grad = False
        for param in self._model.cluster_probe.parameters():
            param.requires_grad = False
        for param in self._model.linear_probe.parameters():
            param.requires_grad = False

        self.device = device
        self._model.to(self.device)
        self._model.train()

        self.criterion = InfoNCE()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=1e-4)

        self.img_buffer = []

        # self.buffer = DataBuffer(max_size=500)

        normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._transform = T.Compose(
            [
                T.Resize(224, T.InterpolationMode.NEAREST),
                T.CenterCrop(224),
                normalization,
            ]
        )

        self._min_samples_for_training = 50

    def forward(self, x):
        # x实际是一个batch的img
        resized_img = self._transform(x).to(self.device)  # 1 3 224 224
        self._code = self._model.get_code(resized_img)
        return self._code

    # def update_buffer(self, img, mask):
    #     feats = self.forward(img)
    #     self.buffer.generate_data(feats[0], mask)

    def make_batch(self, batch_size=8):
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

    def batch_to_data(self, img_batch, masks):
        # step2: 根据img和mask生成正负样本，并判断是否达到最小训练数据量"""
        feats = self.forward(img_batch)

        positive_data = []
        negative_data = []

        for i, mask in enumerate(masks):
            positive_indices = np.where(mask == 1)  # 掩膜为1的位置
            negative_indices = np.where(mask == 0)

            # 转置后形状为[num_positive, 90]和[num_negative, 90]
            positive = feats[i][:, positive_indices[0], positive_indices[1]]
            negative = feats[i][:, negative_indices[0], negative_indices[1]]
            positive_data.append(positive.T)
            negative_data.append(negative.T)
            # print(f"Positive data size: {len(positive.T)}, Negative data size: {len(negative.T)}")

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

    # def lora_train_once(self, img_path, ann_path):
    #     # 选图片、生成数据、一次性训练
    #     self._model.train()
    #     running_loss = 0.0
    #     img, mask = self.select_img(img_path, ann_path)
    #     positive_data, negative_key = self.img_to_data(img, mask)
    #     if len(positive_data) % 2 != 0:
    #         positive_data = positive_data[:-1]  # 去除一个样本以确保长度为偶数
    #
    #     random.shuffle(positive_data)
    #     half_size = len(positive_data) // 2
    #     query = positive_data[:half_size]
    #     positive_key = positive_data[half_size:]
    #
    #     loss = self.criterion(query, positive_key, negative_key)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     running_loss += loss.item()
    #     print(f"Loss: {running_loss}")

    def lora_train(self, num_epoch=10):
        self._model.train()
        num_batches = 3
        for epoch in range(num_epoch):
            running_loss = 0.0
            for _ in range(num_batches):
                img_bath, masks = self.make_batch()
                query, positive_key, negative_key = self.batch_to_data(img_bath, masks)

                loss = self.criterion(query, positive_key, negative_key)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {running_loss / num_batches}")


# def train(model, buffer, criterion, optimizer, device, num_epoch=10, batch_size=4):
#     model.train()
#     for epoch in range(num_epoch):
#         running_loss = 0.0
#         num_batches = len(buffer) // batch_size  # TODO 这里也有问题
#         for _ in range(num_batches):
#             query, positive_keys = buffer.sample_query_and_positive(batch_size=25)
#             negative_keys = buffer.sample_negative(size=50)
#
#             # TODO 这里还得再看看对不对
#             query = torch.stack(query).to(device)
#             positive_keys = torch.stack(positive_keys).to(device)
#             negative_keys = torch.stack(negative_keys).to(device)
#
#             query_emb = model(query)
#             positive_emb = model(positive_keys)
#             negative_emb = model(negative_keys)
#
#             query_emb = torch.flatten(query_emb, start_dim=1)
#             positive_emb = torch.flatten(positive_emb, start_dim=1)
#             negative_emb = torch.flatten(negative_emb, start_dim=1)
#
#             loss = criterion(query_emb, positive_emb, negative_emb)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#         print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {running_loss / num_batches}")

        # for batch_idx, batch in enumerate(dataloader):
        #     query, positive_key, negative_key = batch
        #     query = query.to(device)
        #     positive_key = positive_key.to(device)
        #     negative_key = negative_key.to(device)
        #
        #     query_emb = model(query)
        #     positive_emb = model(positive_key)
        #     negative_emb = model(negative_key)
        #
        #     # 展平嵌入向量
        #     query_emb = torch.flatten(query_emb, start_dim=1)
        #     positive_emb = torch.flatten(positive_emb, start_dim=1)
        #     negative_emb = torch.flatten(negative_emb, start_dim=1)
        #
        #     loss = criterion(query_emb, positive_emb, negative_emb)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     running_loss += loss.item()
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")


if __name__ == "__main__":
    # Normalization and transformation for the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
    model_path = "/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt"
    model = DINOWithLoRA(model_path, device)

    model.lora_train(num_epoch=30)

    # save
    old_ckpt = torch.load(model_path, map_location='cuda:0')
    new_model_state_dict = model._model.state_dict()  # 要取到_model，不然测试时索引失败
    old_ckpt["state_dict"] = new_model_state_dict   # 老的ckpt是用pl模块save的，这里取巧直接把state_dict替换一下保存
    save_path = f"/home/jack/wvn/self_supervised_segmentation/models/lora_ft_1220.ckpt"
    torch.save(old_ckpt, save_path)

    # train_dir = "/home/jack/wvn/SurgicalDINO/data"
    # train_dataset = ContrastiveDataset(train_dir, transform)
    # # num_workers不设置为0会warning: Producer process has been terminated before all shared CUDA tensors released.
    # train_dataloader = DataLoader(
    #                     train_dataset,
    #                     batch_size=4,
    #                     shuffle=True,
    #                     num_workers=0,)
