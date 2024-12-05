
from loraDino.feature_extractor.segment_extractor import SegmentExtractor
from loraDino.feature_extractor.stego_interface import StegoInterface
import torch
import numpy as np
import kornia
from kornia.feature import DenseSIFTDescriptor
from kornia.contrib import extract_tensor_patches, combine_tensor_patches
from fast_slic import Slic


class FeatureExtractor:
    def __init__(
        self,
        device: str,
        segmentation_type: str = "stego",
        feature_type: str = "stego",
        input_size: int = 224,
        **kwargs,
    ):
        """Feature extraction from image

        Args:
            device (str): Compute device
            extractor (str): Extractor model: stego, dino_slic
        """

        self._device = device
        self._segmentation_type = segmentation_type
        self._feature_type = feature_type
        self._input_size = input_size
        self._stego_features_already_computed_in_segmentation = False

        # Prepare segment extractor
        self.segment_extractor = SegmentExtractor().to(self._device)

        # Prepare extractor depending on the type
        self._feature_dim = 90
        self._extractor = StegoInterface(
            device=device,
            input_size=input_size,
            n_image_clusters=kwargs.get("n_image_clusters", 20),
            run_clustering=kwargs.get("run_clustering", True),
            run_crf=kwargs.get("run_crf", False),
        )

        # Segmentation
        self.slic = Slic(
            num_components=kwargs.get("slic_num_components", 50),
            compactness=kwargs.get("slic_compactness", 10),
        )
        # Segmentation
        # from fast_slic import Slic
        #
        # self.slic = Slic(
        #     num_components=kwargs.get("slic_num_components", 100),
        #     compactness=kwargs.get("slic_compactness", 10),
        # )

    def extract(self, img, **kwargs):
        # Compute segments, their centers, and edges connecting them (graph structure)
        # with Timer("feature_extractor - compute_segments"):
        edges, seg, center, linear_seg, seg_slic = self.compute_segments(img, **kwargs)
        # Compute features
        # with Timer("feature_extractor - compute_features"):
        dense_feat = self.compute_features(img, seg, center, **kwargs)
        # with Timer("feature_extractor - compute_features"):
        # Sparsify features to match the centers if required
        feat = self.sparsify_features(dense_feat, seg)

        if kwargs.get("return_dense_features", False):
            return edges, feat, seg, center, dense_feat, linear_seg, seg_slic

        return edges, feat, seg, center, None, linear_seg, seg_slic

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def segmentation_type(self):
        return self._segmentation_type

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._device = device
        self._extractor.change_device(device)

    def compute_segments(self, img: torch.tensor, **kwargs):
        if self._segmentation_type == "stego":
            seg, linear_seg = self.segment_stego(img, **kwargs)
        else:
            raise f"segmentation_type [{self._segmentation_type}] not supported"

        # Compute edges and centers
        if self._segmentation_type != "none" and self._segmentation_type is not None:
            # Extract adjacency_list based on segments
            edges = self.segment_extractor.adjacency_list(seg)
            # Extract centers
            centers = self.segment_extractor.centers(seg)

        seg_slic = self.segment_slic(img, **kwargs)
        return edges.T, seg[0, 0], centers, linear_seg[0, 0], seg_slic[0, 0]

    def segment_slic(self, img, **kwargs):
        # Get slic clusters
        img_np = kornia.utils.tensor_to_image(img)
        seg = self.slic.iterate(np.uint8(np.ascontiguousarray(img_np) * 255))[None, None]
        # return torch.from_numpy(seg).to(self._device).type(torch.long)
        return seg

    def segment_stego(self, img, **kwargs):
        # Prepare input image
        img_internal = img.clone()
        self._extractor.inference(img_internal)
        seg = self._extractor.cluster_segments.to(self._device)
        linear_seg = self._extractor.linear_segments.to(self._device)
        # seg = torch.from_numpy(self._extractor.cluster_segments).to(self._device)

        # Change the segment indices by numbers from 0 to N
        for i, k in enumerate(seg.unique()):
            seg[seg == k.item()] = i

        self._stego_features_already_computed_in_segmentation = True
        return seg, linear_seg

    def compute_features(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        if self._feature_type == "stego":
            feat = self.compute_stego(img, seg, center, **kwargs)
        else:
            raise f"segmentation_type [{self._segmentation_type}] not supported"

        return feat

    @torch.no_grad()
    def compute_stego(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        if self._stego_features_already_computed_in_segmentation:
            self._stego_features_already_computed_in_segmentation = False
            return self._extractor.features
        else:
            img_internal = img.clone()
            self._extractor.inference(img_internal)
            return self._extractor.features

    def sparsify_features(self, dense_features: torch.tensor, seg: torch.tensor, cumsum_trick=False):
        if self._feature_type not in ["histogram"] and self._segmentation_type not in ["none"]:
            # Get median features for each cluster

            if type(dense_features) == dict:
                # Multiscale feature pyramid extraction
                scales_x = [feat.shape[2] / seg.shape[0] for feat in dense_features.values()]
                scales_y = [feat.shape[3] / seg.shape[1] for feat in dense_features.values()]

                segs = [
                    torch.nn.functional.interpolate(
                        seg[None, None, :, :].type(torch.float32),
                        scale_factor=(scale_x, scale_y),
                    )[0, 0].type(torch.long)
                    for scale_x, scale_y in zip(scales_x, scales_y)
                ]
                sparse_features = []

                # Iterate over each segment
                for i in range(seg.max() + 1):
                    single_segment_feature = []

                    # Iterate over each scale
                    for dense_feature, seg_scaled in zip(dense_features.values(), segs):
                        m = seg_scaled == i
                        prev_scale_x = 1.0
                        prev_scale_y = 1.0
                        prev_x = 1.0
                        prev_y = 1.0

                        # When downscaling the mask it becomes 0 therfore calculate x,y
                        # Based on the previous scale
                        if m.sum() == 0:
                            x = (
                                (prev_x * seg_scaled.shape[0] / prev_scale_x)
                                .type(torch.long)
                                .clamp(0, seg_scaled.shape[0] - 1)
                            )
                            y = (
                                (prev_y * seg_scaled.shape[1] / prev_scale_y)
                                .type(torch.long)
                                .clamp(0, seg_scaled.shape[1] - 1)
                            )
                            feat = dense_feature[0, :, x, y]
                        else:
                            x, y = torch.where(m)
                            prev_x = x.type(torch.float32).mean()
                            prev_y = y.type(torch.float32).mean()
                            prev_scale_x = seg_scaled.shape[0]
                            prev_scale_y = seg_scaled.shape[0]
                            feat = dense_feature[0, :, x, y].mean(dim=1)

                        single_segment_feature.append(feat)

                    single_segment_feature = torch.cat(single_segment_feature, dim=0)
                    sparse_features.append(single_segment_feature)
                return torch.stack(sparse_features, dim=1).T

            else:
                if cumsum_trick:
                    # Cumsum is slightly slower for 100 segments
                    # Trick: sort the featuers according to the segments and then use cumsum for summing
                    dense_features = dense_features[0].permute(1, 2, 0).reshape(-1, dense_features.shape[1])
                    seg = seg.reshape(-1)
                    sorts = seg.argsort()
                    dense_features_sort, seg_sort = dense_features[sorts], seg[sorts]
                    x = dense_features_sort
                    # The cumsum operation is the only one that takes times
                    x = x.cumsum(0)
                    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
                    elements_sumed = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
                    kept[:-1] = seg_sort[1:] != seg_sort[:-1]
                    x = x[kept]
                    x = torch.cat((x[:1], x[1:] - x[:-1]))

                    elements_sumed = elements_sumed[kept]
                    elements_sumed[1:] = elements_sumed[1:] - elements_sumed[:-1]
                    x /= elements_sumed[:, None]
                    return x
                else:
                    sparse_features = []
                    for i in range(seg.max() + 1):
                        m = seg == i
                        x, y = torch.where(m)
                        feat = dense_features[0, :, x, y].mean(dim=1)
                        sparse_features.append(feat)
                    return torch.stack(sparse_features, dim=1).T
        else:
            return dense_features
