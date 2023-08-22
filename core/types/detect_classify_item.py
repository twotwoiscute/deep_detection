
from typing import List, Optional
from proto_build.vision_interfaces_pb2 import DetectClassifyItemResponse
from .detection_error_code import TritonDetectionErrorCodes
from .detection_result import TritonDetectionResult

__all__ = [
    "TritonDetectClassifyItemResponse"
]

class TritonDetectClassifyItemResponse:
    def __init__(
        self, 
        error_code: Optional[TritonDetectionErrorCodes],
        detection_results: List[TritonDetectionResult]
    ):
        self.error_code = error_code
        self.detection_results = detection_results
            
    def Triton2gRPC(self):
        grpc_msg = DetectClassifyItemResponse()
        if self.error_code is not None:
            grpc_msg.error_code.CopyFrom(self.error_code.Triton2gRPC())
        if self.detection_results is not None:
            grpc_msg.detection_result_array.extend([
                detection_result.Triton2gRPC()
                for detection_result in self.detection_results]
            )
        return grpc_msg