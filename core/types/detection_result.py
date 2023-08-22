from proto_build.vision_interfaces_pb2 import DetectionResult
from typing import Optional 

from .item_classification_info import TritonItemClassificationInfo
from .region_of_interest import TritonRegionOfInterest
from .image import TritonMask
from .contour import TritonContours

__all__ = [
    "TritonDetectionResult", 
]

class TritonDetectionResult:
    def __init__(
        self, 
        item_classification_info: Optional[TritonItemClassificationInfo] = None,
        bbox: Optional[TritonRegionOfInterest] = None,
        mask: Optional[TritonMask] = None,
        contours: Optional[TritonContours] = None
    ):
        self.item_classification_info = item_classification_info
        self.bbox = bbox
        self.mask = mask
        self.contours = contours

    def Triton2gRPC(self):
        detection_result = DetectionResult()
        if self.item_classification_info is not None:
            detection_result.classification.CopyFrom(self.item_classification_info.Triton2gRPC())
        if self.bbox is not None:
            detection_result.bbox.CopyFrom(self.bbox.Triton2gRPC())
        if self.mask is not None:
            detection_result.mask.CopyFrom(self.mask.Triton2gRPC())
        if self.contours is not None:
            detection_result.contours.extend(self.contours.Triton2gRPC())
        return detection_result
