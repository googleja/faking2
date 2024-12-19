import os
import torch
from tools.FastSAM.wvn.vae import batch_size
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from loraDino.stego.stego import Stego
from loraDino.net_ft.infoNCE import InfoNCE
# from loraDino.net_ft.dataset_gen import ContrastiveDataset
from loraDino.net_ft.dataset_gen import DataBuffer, generate_data, extract_mask
import cv2
import numpy as np

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


class DINOWithLoRA(nn.Module):
    def __init__(self, model_path):
        super(DINOWithLoRA, self).__init__()
        self._model = Stego.load_from_checkpoint(model_path, n_image_clusters=20, strict=False)

        # freeze stego额外层的参数
        for param in self._model.segmentation_head.parameters():
            param.requires_grad = False
        for param in self._model.cluster_probe.parameters():
            param.requires_grad = False
        for param in self._model.linear_probe.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x实际是一个batch的img
        self._code = self._model.get_code(x)
        return self._code


def train(model, buffer, criterion, optimizer, device, num_epoch=10, batch_size=4):
    model.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        num_batches = len(buffer) // batch_size
        for _ in range(num_batches):
            query, positive_keys = buffer.sample_query_and_positive(batch_size=25)
            negative_keys = buffer.sample_negative(batch_size=50)

            #TODO 这里还得再看看对不对
            query = torch.stack(query).to(device)
            positive_keys = torch.stack(positive_keys).to(device)
            negative_keys = torch.stack(negative_keys).to(device)

            query_emb = model(query)
            positive_emb = model(positive_keys)
            negative_emb = model(negative_keys)

            query_emb = torch.flatten(query_emb, start_dim=1)
            positive_emb = torch.flatten(positive_emb, start_dim=1)
            negative_emb = torch.flatten(negative_emb, start_dim=1)

            loss = criterion(query_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {running_loss / num_batches}")

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
    normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.NEAREST),
            T.CenterCrop(224),
            normalization,
        ]
    )

    # prepare data
    buffer = DataBuffer(max_size=100)
    test_img = "/home/jack/wvn/SurgicalDINO/data1219/image/trail-15_00361.png"
    test_ann = "/home/jack/wvn/SurgicalDINO/data1219/annotation/trail-15_00361.png"
    mask = extract_mask(test_ann, test_img, target_range=((240, 220, 195), (255, 235, 215)))
    mask_resized = cv2.resize(mask.astype(np.uint8), (28, 28), interpolation=cv2.INTER_NEAREST)

    features = np.random.randn(90, 28, 28)  # 模拟Stego提取的特征
    positive_samples, negative_samples = generate_data(features, mask_resized)

    buffer.add_positive(positive_samples)
    buffer.add_negative(negative_samples)

    # prepare model
    model_path = "/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt"
    model = DINOWithLoRA(model_path)
    criterion = InfoNCE()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model.to(device)

    # train
    if buffer.is_ready(50, 100):
        train(model, buffer, criterion, optimizer, device, num_epoch=5, batch_size=4)

    # save
    old_ckpt = torch.load(model_path, map_location='cuda:0')
    new_model_state_dict = model._model.state_dict()  # 要取到_model，不然测试时索引失败
    old_ckpt["state_dict"] = new_model_state_dict   # 老的ckpt是用pl模块save的，这里取巧直接把state_dict替换一下保存
    save_path = f"/home/jack/wvn/self_supervised_segmentation/models/lora_ft_1216.ckpt"
    torch.save(old_ckpt, save_path)

    # train_dir = "/home/jack/wvn/SurgicalDINO/data"
    # train_dataset = ContrastiveDataset(train_dir, transform)
    # # num_workers不设置为0会warning: Producer process has been terminated before all shared CUDA tensors released.
    # train_dataloader = DataLoader(
    #                     train_dataset,
    #                     batch_size=4,
    #                     shuffle=True,
    #                     num_workers=0,)
