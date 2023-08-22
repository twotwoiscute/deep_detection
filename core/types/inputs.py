from typing import List, Optional
import numpy as np
from loguru import logger

from proto_build.camera3d_interfaces_pb2 import RGBDCameraData
from .region_of_interest import TritonRegionOfInterest
from ..utils import raise_exception_error


__all__ = [
    "TritonDetectClassifyItemRequest", 
    "TritonRGBDCameraData", 
    "TritonDetectClassifyItemRequest"
]

DETECTION_TYPES_LSIT = [0, 1, 2, 3, 4, 100, 101]
name_to_dtypes = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),
    # OpenCV CvMat types
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4),
}
    
class TritonRGBDCameraData:
    def __init__(
        self, 
        color_data: Optional[np.ndarray] = None,
        depth_data: Optional[np.ndarray] = None,
    ):
        self.color_data = color_data
        self.depth_data = depth_data      
    
    @classmethod
    def gRPC2Triton(cls, grpc_camera_data: RGBDCameraData):
        color_numpy_data = None
        depth_numpy_data = None
        if grpc_camera_data.color.height:
            try:
                color_dtype, num_color_channels = name_to_dtypes[grpc_camera_data.color.encoding]
                color_numpy_data = np.frombuffer(grpc_camera_data.color.data, dtype=color_dtype).reshape(
                                            grpc_camera_data.color.height,
                                            grpc_camera_data.color.width,
                                            num_color_channels
                )
            except:
                raise_exception_error(
                    message="Fail to get color data",
                    logger=logger,
                    exception_error=EOFError
                )

        if grpc_camera_data.depth.height:
            try:
                depth_encoding = grpc_camera_data.depth.encoding
                depth_dtype, num_depth_channels = name_to_dtypes[depth_encoding]
                depth_numpy_data = np.frombuffer(grpc_camera_data.depth.data, dtype=depth_dtype).reshape(
                                            grpc_camera_data.color.height,
                                            grpc_camera_data.color.width,
                                            num_depth_channels
                ) / 1000.0
            except:
                raise_exception_error(
                    message="Fail to get depth data",
                    logger=logger,
                    exception_error=EOFError
                )
        
        return cls(color_numpy_data, depth_numpy_data) 

class TritonDetectionDataTypes:
    def __init__(self, data_types: List[int]):
        self.data_types = data_types
    
    @classmethod
    def gRPC2Triton(cls, data_types):
        data_types_list = list(data_types.types)
        if len(set(data_types_list)) != len(data_types_list):
            logger.warning("Duplicate data_types: {}".format(data_types_list))
            data_types_list = list(set(data_types_list))

        for data_type in data_types_list:
            if data_type not in data_types_list:
                raise_exception_error(
                    message="Invalid data_type, should be one of {} (got {})".format(
                        DETECTION_TYPES_LSIT, data_type
                    ),
                    logger=logger,
                    exception_error=ValueError
                )
        return cls(data_types_list)

class TritonDetectClassifyItemRequest:
    def __init__(
        self,
        camera_data: Optional[TritonRGBDCameraData] = None,
        roi: Optional[TritonRegionOfInterest] = None,
        detection_data_types: Optional[TritonDetectionDataTypes] = None
    ):
        self.camera_data = camera_data
        self.roi = roi
        self.detection_data_types = detection_data_types
    
    @classmethod
    def gRPC2Triton(cls, request):
        camera_data = TritonRGBDCameraData.gRPC2Triton(request.camera_data)
        roi = TritonRegionOfInterest.gRPC2Triton(request.roi)
        data_types = TritonDetectionDataTypes.gRPC2Triton(request.data_types)
        return cls(
            camera_data=camera_data,
            roi=roi,
            detection_data_types=data_types
        )