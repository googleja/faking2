
from loraDino.feature_extractor.feature_extractor import FeatureExtractor
from loraDino.stego.data import create_cityscapes_colormap as cmap

import torch
import numpy as np
import torch.nn.functional as F
import traceback
from skimage.segmentation import mark_boundaries
from PIL import Image
import cv2
from torchvision import transforms as T


"""用来测试wvn最原始的ckpt，唯一区别在于我把它提供的ckpt refactor成了zzy_refactor.ckpt,
同时修改了n_image_clusters，目的是让wvn暴露出分类错误的问题"""


def img_to_torch(img_path, device="cuda"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_tensor = T.ToTensor()
    return to_tensor(img).to(device)


class WvnFeatureExtractor:
    def __init__(self):
        # Setup modules. Param value:cuda,stego,stego,8,vit_small,224,100
        self._feature_extractor = FeatureExtractor(
            'cuda',
            segmentation_type='stego',
            feature_type='stego',
            patch_size=8,
            backbone_type='vit_small',
            input_size=224,
            slic_num_components=50,
            model_path=f"/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt",
            n_image_clusters=10,
        )

    def image_process(self, torch_image):
        # Extract features, feat是稀疏的，dense_feat是密集的
        _, feat, seg, center, dense_feat, linear_seg, seg_slic = self._feature_extractor.extract(
            img=torch_image[None],
            return_centers=False,
            return_dense_features=True,
            n_random_pixels=100,
        )

        return seg, linear_seg, seg_slic


# def test_origin_stego_1():
#     wvn = WvnFeatureExtractor()
#     # dir = '/home/jack/wvn/jackal_img_backup/'
#     dir = '/home/jack/wvn/SurgicalDINO/jackal_img/'
#     for i in range(1, 107):
#         try:
#             img_dir = dir + str(i) + '_torch.pt'
#             torch_image = torch.load(img_dir).to('cuda')
#             seg, linear_seg, seg_slic = wvn.image_process(torch_image)
#
#             # save cluster seg
#             color = cmap()
#             seg_img = color[seg.cpu() % color.shape[0]]
#             seg_img_pil = Image.fromarray((seg_img * 255).astype(np.uint8))
#             seg_img_pil.save(dir + str(i) + "_cluster.png")
#
#             # save linear seg
#             color = cmap()
#             linear_seg_img = color[linear_seg.cpu()]
#             linear_seg_img_pil =Image.fromarray((linear_seg_img * 255).astype(np.uint8))
#             linear_seg_img_pil.save(dir + str(i) + "_linear.png")
#
#             # save slic seg
#             np.save(dir + str(i) + "_slic.npy", seg_slic)
#             np_img = (torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
#             boundaries = mark_boundaries(np_img, seg_slic, color=(0, 0, 1))
#             boundaries_img = Image.fromarray((boundaries * 255).astype(np.uint8))
#             boundaries_img.save(dir + str(i) + "_boundaries.png")
#
#         except Exception as e:
#             print(e)
#             break


def test_origin_stego():
    wvn = WvnFeatureExtractor()

    dir = '/home/jack/wvn/SurgicalDINO/data1219/image/'
    torch_image = img_to_torch(dir + "trail-15_00361.png")

    seg, linear_seg, seg_slic = wvn.image_process(torch_image)

    # save cluster seg
    color = cmap()
    seg_img = color[seg.cpu() % color.shape[0]]
    seg_img_pil = Image.fromarray((seg_img * 255).astype(np.uint8))
    seg_img_pil.save(dir + "_cluster.png")

    # save linear seg
    color = cmap()
    linear_seg_img = color[linear_seg.cpu()]
    linear_seg_img_pil = Image.fromarray((linear_seg_img * 255).astype(np.uint8))
    linear_seg_img_pil.save(dir + "_linear.png")


if __name__ == "__main__":
    test_origin_stego()
