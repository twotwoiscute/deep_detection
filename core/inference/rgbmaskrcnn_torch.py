from typing import List, Dict, Union
import numpy as np
import torch
from detectron2.structures import Instances

class RGBMaskRCNNTorchInfer:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_image: List[Dict[str, Union["torch.tensor", int]]]) -> Instances:
        if isinstance(input_image[0]["image"], np.ndarray):
            input_image[0]["image"] = torch.from_numpy(input_image[0]["image"])
        return self.model(input_image)[0]["instances"].to("cpu")
