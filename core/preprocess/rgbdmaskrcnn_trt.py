import sys
from typing import List, Optional
import numpy as np 
from PIL import Image
from loguru import logger
from ..utils import raise_exception_error

class RGBDMaskRCNNTensorRTPreprocess:
    def __init__(
        self, 
        color_img: np.ndarray,
        depth_img: np.ndarray, 
        short_length:int, 
        max_size:int = sys.maxsize, 
        depth_min:float = 1.0,
        depth_max:float = 1.5, 
        stride:int = 32, 
        rois: Optional[List[int]] = None
    ):
        if rois is not None:
            raise_exception_error(
                message="The region_of_interest(roi) feature is currently not supported.", 
                logger=logger,
                exception_error=NotImplementedError
            )
        
        self.color_image = color_img
        if self.color_image is None:
            raise_exception_error(
                message="Color image is required.",
                logger=logger,
                exception_error=FileNotFoundError
            )
        self.depth_image = depth_img
        if self.depth_image is None:
            raise_exception_error(
                message="Depth image is required",
                logger=logger,
                exception_error=FileNotFoundError
            )            
        self.short_length = short_length
        self.max_size = max_size
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.stride = stride
        self.ori_height, self.ori_width, _ = depth_img.shape

    def normalizedepthimg(self):
        self.depth_image[self.depth_image > self.depth_max] = 0
        self.depth_image -= self.depth_min
        self.depth_image[self.depth_image < 0] = 0
        min_depth = np.min(self.depth_image)
        cropped_depth_image = (self.depth_image - min_depth) / (self.depth_max - self.depth_min)
        self.depth_image = np.uint8(cropped_depth_image * 255)
       
    def set_output_hw(self):
        h, w = self.ori_height, self.ori_width
        size = np.random.randint(self.short_length, self.short_length + 1)
        
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        self.resize_w = int(neww + 0.5)
        self.resize_h = int(newh + 0.5)

    def resize(self, img, interp=Image.BILINEAR):
        if img.shape[:2] != (self.ori_height, self.ori_width):
            logger.warning(
                "The shape has already changed before resizeing, expected {}, got{}".format(
                    img.shape[:2], (self.ori_height, self.ori_width)
                )
            )
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((self.resize_w, self.resize_h), interp)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)

        return ret
    
    def get_input_image(self) -> np.ndarray:
        self.set_output_hw()
        self.normalizedepthimg()
        self.color_image = self.resize(self.color_image)
        self.depth_image = self.resize(self.depth_image)
        # cat 
        cat_image = np.concatenate([self.color_image, self.depth_image], axis=2)
        image = np.asarray(cat_image, dtype=np.float32)

        # calculate padding 
        height, width, _ = self.color_image.shape
        pad_x = int(np.ceil(width / self.stride) * self.stride - width)
        pad_y = int(np.ceil(height / self.stride) * self.stride - height)
        # Change HWC -> CHW.
        image = np.transpose(image, (2, 0, 1))[None]
        return (image, (pad_x, pad_y))
