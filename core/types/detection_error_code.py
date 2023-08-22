from proto_build.vision_interfaces_pb2 import DetectionErrorCodes

__all__ = [
    "TritonDetectionErrorCodes"
]

class TritonDetectionErrorCodes:
    def __init__(self, value: int):
        self.value = value
    def Triton2gRPC(self):
        detection_error_codes = DetectionErrorCodes(value=self.value)
        return detection_error_codes