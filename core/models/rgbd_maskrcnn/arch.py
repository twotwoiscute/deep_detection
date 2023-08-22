import torch

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList

__all__ = ["RGBDRCNN"]


@META_ARCH_REGISTRY.register()
class RGBDRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        print("\nInitializing RGBDRCNN\n")
        super().__init__(cfg)

        depth_pixel_mean = cfg.MODEL.DEPTH_PIXEL_MEAN
        depth_pixel_std = cfg.MODEL.DEPTH_PIXEL_STD
        self.register_buffer("depth_pixel_mean", torch.Tensor(depth_pixel_mean).view(-1, 1, 1))
        self.register_buffer("depth_pixel_std", torch.Tensor(depth_pixel_std).view(-1, 1, 1))
        assert (
            self.depth_pixel_mean.shape == self.depth_pixel_std.shape
        ), f"{self.depth_pixel_mean} and {self.depth_pixel_std} have different shapes!"

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for x in batched_inputs:
            x = x["image"].to(self.device)
            x[0:3, :, :] = (x[0:3, :, :] - self.pixel_mean) / self.pixel_std
            x[3, :, :] = (x[3, :, :] - self.depth_pixel_mean) / self.depth_pixel_std
            images.append(x)

        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
