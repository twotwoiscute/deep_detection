from typing import List
import numpy as np
from proto_build.vision_interfaces_pb2 import Contour
from .pixel import TritonPixel

__all__ = [
    "TritonContour", 
    "TritonContours",
]

class TritonContour:
    def __init__(self, contour: np.ndarray):
        self.contour = contour

    def Triton2gRPC(self):
        contour = Contour(
            contour=[TritonPixel(int(p[0]), int(p[1])).Triton2gRPC()
            for p in self.contour]
        )
        return contour

class TritonContours:
    def __init__(self, contours: List[TritonContour]):
        self.contours = contours

    def Triton2gRPC(self) -> List[TritonContour]:
        return [contour.Triton2gRPC() for contour in self.contours]