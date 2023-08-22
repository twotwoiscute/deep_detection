import numpy as np
from typing import Optional
from fvcore.transforms.transform import Transform

from detectron2.data.transforms import StandardAugInput

__all__ = [
    "RGBDAugInput",
]


class RGBDAugInput(StandardAugInput):
    def __init__(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        super().__init__(image=rgb_image, boxes=boxes, sem_seg=sem_seg)
        assert isinstance(
            depth_image, np.ndarray
        ), "[Augmentation] Needs an numpy array, but got a {}!".format(type(depth_image))
        assert not np.issubdtype(depth_image.dtype, np.floating) or (
            depth_image.dtype == np.uint8
        ), "[Augmentation] Got image of typr {}, use uint8 or floating points instead!".format(
            depth_image.dtype
        )
        self.depth_image = depth_image

    def transform(self, tfm: Transform) -> None:
        self.depth_image = tfm.apply_image(self.depth_image)
        super().transform(tfm)
