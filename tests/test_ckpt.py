
from loraDino.feature_extractor.feature_extractor import FeatureExtractor
from loraDino.stego.data import create_cityscapes_colormap as cmap

import torch
import numpy as np
import torch.nn.functional as F
import traceback
from skimage.segmentation import mark_boundaries
from PIL import Image

# 基本思路：
# 1、它是多机位相机，所以使用了scheduler根据优先级选择当前应当处理哪一帧图像
# 2、图像改为224x224，经由STEGO提取90维dense feature和segmentation mask
# 3、将224x224x90的dense feature降采样到100个embeddings（也是用的STEGO） NOTE 实际好像没降到100
# 4、输入到MLP中，得到traversability map。MLP由learning node在线训练

# 也就是它的特征提取完全依赖DINO，分割依靠STEGO，并没有作任何训练，只是后续用MLP作训练来预测traversability map


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
            model_path=f"/home/jack/wvn/self_supervised_segmentation/models/lora_ft_1216.ckpt",
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


if __name__ == "__main__":
    wvn = WvnFeatureExtractor()
    # dir = '/home/jack/wvn/jackal_img_backup/'
    dir = '/jackal_img/'
    for i in range(1, 107):
        try:
            img_dir = dir + str(i) + '_torch.pt'
            torch_image = torch.load(img_dir).to('cuda')
            seg, linear_seg, seg_slic = wvn.image_process(torch_image)

            # save cluster seg
            color = cmap()
            seg_img = color[seg.cpu() % color.shape[0]]
            seg_img_pil = Image.fromarray((seg_img * 255).astype(np.uint8))
            seg_img_pil.save(dir + str(i) + "_lora_cluster.png")

            # save linear seg
            color = cmap()
            linear_seg_img = color[linear_seg.cpu()]
            linear_seg_img_pil =Image.fromarray((linear_seg_img * 255).astype(np.uint8))
            linear_seg_img_pil.save(dir + str(i) + "_lora_linear.png")

            # # save slic seg
            # np.save(dir + str(i) + "_slic.npy", seg_slic)
            # np_img = (torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            # boundaries = mark_boundaries(np_img, seg_slic, color=(0, 0, 1))
            # boundaries_img = Image.fromarray((boundaries * 255).astype(np.uint8))
            # boundaries_img.save(dir + str(i) + "_lora_boundaries.png")

        except Exception as e:
            print(e)
            break
