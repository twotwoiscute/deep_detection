from typing import List, Dict, Union
import detectron2.data.transforms as T
import torch
import numpy as np
from loguru import logger 

class RGBMaskRCNNTorchPreprocess:
    def __init__(self, 
        min_size_test: int, 
        max_size_test: int,
        rois: np.ndarray = None
    ): 
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.augs = T.ResizeShortestEdge(
            [self.min_size_test, self.min_size_test],
            self.max_size_test
        )
        self.rois = rois
    
    def get_input_image(self, input: np.ndarray) -> List[Dict[str, Union[int, np.ndarray]]]:
        if self.rois is not None: 
            xmin, ymin, width, height = self.rois
            input = input[ymin: ymin + height, xmin: xmin + width]
        ori_height, ori_width, _ = input.shape
        input = self.augs.get_transform(input).apply_image(input)
        input = torch.as_tensor(input.transpose(2, 0, 1), dtype=torch.float32)
        inputs = {
            "image": input,
            "height": ori_height,
            "width": ori_width,
        }
        return [inputs]
