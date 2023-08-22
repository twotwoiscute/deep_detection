import sys
from typing import List, Optional
import numpy as np 
from PIL import Image
from loguru import logger
from ..utils import raise_exception_error
from .rgbdmaskrcnn_trt import RGBDMaskRCNNTensorRTPreprocess

class RGBDMaskRCNNTorchPreprocess(RGBDMaskRCNNTensorRTPreprocess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)