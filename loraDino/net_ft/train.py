
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import pytorch_lightning as pl

from loraDino.stego.stego import Stego
from loraDino.net_ft.infoNCE import InfoNCE
from loraDino.net_ft.dataset_gen import ContrastiveDataset


class DINOWithLoRA(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()
        self._model = Stego.load_from_checkpoint(model_path, n_image_clusters=20, strict=False)
        self._model.eval()

        # normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # self._transform = T.Compose(
        #     [
        #         T.Resize(224, T.InterpolationMode.NEAREST),
        #         T.CenterCrop(224),
        #         normalization,
        #     ]
        # )

        # freeze stego额外层的参数
        for param in self._model.segmentation_head.parameters():
            param.requires_grad = False
        for param in self._model.cluster_probe.parameters():
            param.requires_grad = False
        for param in self._model.linear_probe.parameters():
            param.requires_grad = False

        # 查看哪些层会参与训练
        # for name, param in self._model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: requires_grad={param.requires_grad}")

        self.criterion = InfoNCE()

    def forward(self, img):
        # resized_img = self._transform(img)
        self._code = self._model.get_code(img)
        return self._code

    def training_step(self, batch, batch_idx):
        self._model.eval()
        query, positive_key, negative_keys = batch
        query_emb = self._model(query)
        positive_emb = self._model(positive_key)
        negative_emb = self._model(negative_keys)

        #TODO 这里的emb还没有展平

        loss = self.criterion(query_emb, positive_emb, negative_emb)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    model_path = f"/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt"
    model = DINOWithLoRA(model_path)

    # dir = '/home/jack/wvn/SurgicalDINO/jackal_img/'
    # try:
    #     img_dir = dir + '1_torch.pt'
    #     torch_image = torch.load(img_dir)
    #     code = model.forward(torch_image[None])
    # except Exception as e:
    #     print(e)

    normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.NEAREST),
            T.CenterCrop(224),
            normalization,
        ]
    )
    train_dir = f"/home/jack/wvn/SurgicalDINO/data"
    train_dataset = ContrastiveDataset(train_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.fit(model, train_dataloader)