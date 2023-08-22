import os
import threading 
import json
import time
from loguru import logger

from core.preprocess import RGBDMaskRCNNTensorRTPreprocess
from core.postprocess import RGBDMaskRCNNTensorRTPostprocess
from core.inference import RGBDMaskRCNNTensorRTInfer
from core.types import TritonDetectClassifyItemRequest
from core.utils import reinterpret_roi
from core.utils import raise_exception_error

from proto_build import vision_interfaces_pb2_grpc
from .build import SERVER_REGISTRY



@SERVER_REGISTRY.register()
class RGBDMaskRCNNTensorRTServicer(vision_interfaces_pb2_grpc.DetectClassifyItemServicer):
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
        self.score_threshold = self.config_data["SCORE_THRESHOLD"]
        self.mask_score_threshold = self.config_data["MASK_SCORE_THRESHOLD"]
        dir_name = os.path.dirname(os.path.abspath(node_config_file))
        self.categories = os.path.join(dir_name, self.config_data["CATEGORY_FILE"])
        if not os.path.exists(self.categories):
            raise_exception_error(
                message="Fail to read category file, The category file ought to be placed at the same level of node config file.",
                logger=logger,
                exception_error=FileNotFoundError
            )
        self.engine = os.path.join(dir_name, self.config_data["ENGINE"])
        if not os.path.isfile(self.engine):
            raise_exception_error(
                message="Engine file not found.",
                logger=logger,
                exception_error=FileNotFoundError,
            )
        self.trt_infer = RGBDMaskRCNNTensorRTInfer(self.engine)

    def load_node_config(self, node_config_file):
        with open(node_config_file, "r") as file:
            self.config_data = json.load(file)

    def Call(self, request, context):
        with self.lock:
            self.trt_infer.create_context()
            
            self.outputs = []
            self.inputs = []
            self.preprocess(request)
            t1 = time.time()
            # infer
            self.trt_outputs = self.trt_infer(self.input_image)
            t2 = time.time()
            logger.info("The inference time is {}({}fps)".format(
                int((t2 - t1) * 1000), 
                int(1000 / ((t2 - t1) * 1000))
            ))
            
            vision_response = self.postprocess()
            return vision_response
    
    def preprocess(self, request, preprocessor=None):
        self.input_data = TritonDetectClassifyItemRequest.gRPC2Triton(request)
        rois = reinterpret_roi(self.input_data.roi, self.input_data.camera_data.color_data.shape[:-1][::-1])
        if preprocessor is None:
            preprocessor = RGBDMaskRCNNTensorRTPreprocess(self.input_data.camera_data.color_data,
                                        self.input_data.camera_data.depth_data,
                                        short_length=self.short_edge,
                                        depth_min=self.depth_min,
                                        depth_max=self.depth_max,
                                        rois=rois)
        else:
            preprocessor = preprocessor(color_img=self.input_data.camera_data.color_data,
                                        depth_img=self.input_data.camera_data.depth_data,
                                        short_length=self.short_edge,
                                        depth_min=self.depth_min,
                                        depth_max=self.depth_max,
                                        rois=rois)
                                        
        self.ori_height, self.ori_width = preprocessor.ori_height, preprocessor.ori_width
        self.input_image, self.pad_xy = preprocessor.get_input_image()
        self.cur_height, self.cur_width = self.input_image.shape[2:]

    def postprocess(self):
        # post processing 
        postprocessor = RGBDMaskRCNNTensorRTPostprocess(
                                        cur_hw_shape=(self.cur_height,self.cur_width), 
                                        ori_hw_shape=(self.ori_height, self.ori_width),
                                        category_list=self.categories,
                                        request_data_type=self.input_data.detection_data_types.data_types
                                    )
        
        detection_results = postprocessor. \
            parse_trt_detection_output( 
                self.trt_outputs,
                self.pad_xy, 
                self.score_threshold
            )
        
        result = postprocessor.postprocess(detection_results, self.mask_score_threshold)
        return postprocessor.Triton2gRPC(result)