from proto_build.vision_interfaces_pb2 import Pixel

__all__ = [
    "TritonPixel"
]

class TritonPixel:
    def __init__(self,x:int, y:int):
        self.x = x
        self.y = y

    def Triton2gRPC(self):
        pixel = Pixel(x=self.x, y=self.y)
        return pixel 