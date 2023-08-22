import os
import threading
from typing import List, Union, Optional
import json
import time
from loguru import logger

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model

from core.server import RGBDMaskRCNNTensorRTServicer
from core.preprocess import RGBDMaskRCNNTorchPreprocess
from core.inference import RGBMaskRCNNTorchInfer
from core.postprocess import RGBMaskRCNNTorchPostprocess
from core.types import TritonDetectClassifyItemRequest
from core.utils import reinterpret_roi
from core.utils import raise_exception_error

from proto_build import vision_interfaces_pb2_grpc
from .build import SERVER_REGISTRY

from ..models.rgbd_maskrcnn import *

@SERVER_REGISTRY.register()
class RGBDMaskRCNNTorchServicer(RGBDMaskRCNNTensorRTServicer):
    def __init__(self, node_config_file: str):
        self.lock = threading.Lock()
        if not os.path.exists(node_config_file):
            raise_exception_error(
                message="The config file for detection is not found.",
                logger=logger,
                exception_error=FileNotFoundError
            )
        self.load_node_config(node_config_file)
        self.depth_max = self.config_data["DEPTH_MAX"] / 1000.
        self.depth_min = self.config_data["DEPTH_MIN"] / 1000.
        self.short_edge = self.config_data["MIN_SIZE"]
        self.score_threshold = self.config_data["SCORE_THRESH_TEST"]
        dir_name = os.path.dirname(os.path.abspath(node_config_file))
        self.categories = os.path.join(dir_name, self.config_data["CATEGORY_FILE"])
        if not os.path.exists(self.categories):
            raise_exception_error(
                message="Fail to read category file, The category file ought to be placed at the same level of node config file.",
                logger=logger,
                exception_error=FileNotFoundError
            )

        self.model_cfg = os.path.join(
            dir_name, self.config_data["CONFIG_FILE"]
        )
        if not os.path.exists(self.model_cfg):
            raise_exception_error(
                message="Config for model not found",
                logger=logger,
                exception_error=FileNotFoundError
            )

        self.model_path = os.path.join(dir_name, self.config_data["MODEL_WEIGHT"])
        if not os.path.isfile(self.model_path):
            raise_exception_error(
                message="Model file not found.",
                logger=logger,
                exception_error=FileNotFoundError,
            )
        opts = [
            "INPUT.MIN_SIZE_TEST",
            self.short_edge, 
            "MODEL.WEIGHTS",
            self.model_path,
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 
            self.score_threshold
        ]
        self.cfg = self.setup_cfg(self.model_cfg, opts)
        model = build_model(self.cfg)
        DetectionCheckpointer(model).resume_or_load(self.cfg.MODEL.WEIGHTS)
        model.eval()
        logger.info("Model initialized")
        self.model = RGBMaskRCNNTorchInfer(model)

    def load_node_config(self, node_config_file: str):
        with open(node_config_file, "r") as file:
            self.config_data = json.load(file)

    def setup_cfg(self, 
        config_file: str, 
        opts: List[Union[str, float]],
    ) -> CfgNode:
        # setup
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()
        return cfg

    def Call(self, request, context):  
        with self.lock:
            self.preprocess(request, RGBDMaskRCNNTorchPreprocess)
            # packup the input to feed into detectron2 frame
            
            self.input_image = [dict(
                image=self.input_image.squeeze(), 
                height=self.ori_height,
                width=self.ori_width
            )]
            t1 = time.time()
            # infer
            outputs = self.model(self.input_image)
            t2 = time.time()
            logger.info("The inference time is {}({}fps)".format(
                int((t2 - t1) * 1000), 
                int(1000 / ((t2 - t1) * 1000))
            ))
            
            vision_response = self.postprocess(outputs)
            return vision_response
        
    def postprocess(self, outputs):
        postprocessor = RGBMaskRCNNTorchPostprocess(category_list=self.categories, 
                    request_data_type=self.input_data.detection_data_types.data_types)
        # post processing
        detection_results =postprocessor.parse_detection_output(outputs)
        return postprocessor.Triton2gRPC(detection_results)