import numpy as np 
from proto_build.sensor_msgs_pb2 import Image

__all__ = [
    "TritonMask"
]

class TritonMask:
    def __init__(self, 
                mask: np.ndarray,
                encoding: str = "mono8"):
        self.mask = mask
        self.encoding = encoding

    def Triton2gRPC(self):
        height, width = self.mask.shape
        is_bigendian = self.mask.dtype.byteorder == ">"
        mask = Image(
            height=height,
            width=width,
            encoding=self.encoding, 
            is_bigendian=is_bigendian,
            data=self.mask.tobytes()
        )
        return mask
