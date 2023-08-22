from proto_build.vision_interfaces_pb2 import ItemClassificationInfo

__all__ = [
    "TritonItemClassificationInfo"
]

INT_TO_NAME_MAPPING = {
    0: "NONE",
    1: "BOX",
    2: "ENVELOPE",
    3: "PACKAGE",
    4: "TEXT",
    5: "BARCODE",
    6: "PALLET",
    7: "TOTE",
    8: "PERSON",
    9: "VEHICLE",
    10: "OBSTACLE",
}

NAME_TO_INT_MAPPING = {
    "NONE": 0,
    "BOX": 1,
    "ENVELOPE": 2,
    "PACKAGE": 3,
    "TEXT": 4,
    "BARCODE": 5,
    "PALLET": 6,
    "TOTE": 7,
    "PERSON": 8,
    "VEHICLE": 9,
    "OBSTACLE": 10,
}


class TritonItemClassificationInfo:
    def __init__(self, label: str, score: float):
        self.label = label
        self.score = score
    
    def Triton2gRPC(self):
        item = ItemClassificationInfo(
            type=NAME_TO_INT_MAPPING[self.label],
            score=self.score
        )
        return item