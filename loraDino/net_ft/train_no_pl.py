import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from loraDino.stego.stego import Stego
from loraDino.net_ft.infoNCE import InfoNCE
from loraDino.net_ft.dataset_gen import ContrastiveDataset

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

def train(model, num_epoch, dataloader, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            query, positive_key, negative_key = batch
            query = query.to(device)
            positive_key = positive_key.to(device)
            negative_key = negative_key.to(device)

            query_emb = model(query)
            positive_emb = model(positive_key)
            negative_emb = model(negative_key)

            # 展平嵌入向量
            query_emb = torch.flatten(query_emb, start_dim=1)
            positive_emb = torch.flatten(positive_emb, start_dim=1)
            negative_emb = torch.flatten(negative_emb, start_dim=1)

            loss = criterion(query_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")


if __name__ == "__main__":
    # Normalization and transformation for the dataset
    normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.NEAREST),
            T.CenterCrop(224),
            normalization,
        ]
    )

    train_dir = "/home/jack/wvn/SurgicalDINO/data"
    train_dataset = ContrastiveDataset(train_dir, transform)
    # num_workers不设置为0会warning: Producer process has been terminated before all shared CUDA tensors released.
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0,)

    model_path = "/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt"
    model = DINOWithLoRA(model_path)

    criterion = InfoNCE()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10

    train(model, num_epochs, train_dataloader, criterion, optimizer, device)
