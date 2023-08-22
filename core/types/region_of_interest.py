from proto_build.sensor_msgs_pb2 import RegionOfInterest

__all__ = [
    "TritonRegionOfInterest"
]

class TritonRegionOfInterest:
    def __init__(self, 
                xmin: int,
                ymin: int, 
                height: int, 
                width: int,
                do_rectify: bool = False):
        self.xmin = xmin
        self.ymin = ymin
        self.height = height
        self.width = width
        self.do_rectify = do_rectify
    
    @classmethod
    def gRPC2Triton(
        cls, 
        roi: RegionOfInterest,
        do_rectify: bool = False):
        xmin = int(roi.x_offset)
        ymin = int(roi.y_offset)
        height = int(roi.height)
        width = int(roi.width)
        return cls(
            xmin,
            ymin,
            height,
            width,
            do_rectify
        )

    def Triton2gRPC(self):
        bbox = RegionOfInterest(
                x_offset=self.xmin, 
                y_offset=self.ymin,
                height=self.height, 
                width=self.width,
                do_rectify=self.do_rectify
            )
        return bbox
        