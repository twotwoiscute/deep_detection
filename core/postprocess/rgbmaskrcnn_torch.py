from typing import List, Dict, Any
from detectron2.structures import Instances
from ..utils import reinterpret_request_datatypes, raise_exception_error, _load_category_file
from ..types import TritonItemClassificationInfo, \
                    TritonRegionOfInterest, \
                    TritonMask, \
                    TritonContour, \
                    TritonContours, \
                    TritonDetectionResult, \
                    TritonDetectionErrorCodes, \
                    TritonDetectClassifyItemResponse
import numpy as np
import cv2 as cv
from loguru import logger 

class RGBMaskRCNNTorchPostprocess:
    def __init__(self,
        category_list: List[str],  
        request_data_type: List = [0, 1, 2, 100], 
        batch_size: int = 1,
    ):
        if batch_size != 1: 
            raise_exception_error(
                message="Currently only support single batch.",
                logger=logger,
                exception_error=ValueError
            )
        self.category_list = _load_category_file(category_list)
        self.request_data_type = reinterpret_request_datatypes(request_data_type)
    
    def parse_detection_output(self, outputs: Instances) -> List[TritonDetectionResult]:
        if not len(outputs):
            logger.warning(
                "Either detection result or the request_data_type is empty."
            )
            return []
        
        triton_item, triton_bbox, triton_mask, triton_mask, triton_contour= None, None, None, None, None
        triton_detection_results = []
        boxes: np.ndarray = outputs.get("pred_boxes").tensor.detach().numpy().astype(np.int32)# shape(N, 4) with format(x1, y1, x2, y2)
        pred_classes: np.ndarray = outputs.get("pred_classes").numpy().astype(np.int32) # shape(N,)
        masks: np.ndarray = outputs.get("pred_masks").numpy().astype(np.uint8) # shape(N, ori_h, ori_w)
        scores: np.ndarray = outputs.get("scores").detach().numpy() # float32, shape(N,)
        
        # convert (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height)
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        
        # convert to grpc msg
        for i in range(len(outputs)):
            if self.request_data_type["BBOX"]:
                triton_bbox = TritonRegionOfInterest(
                    xmin=int(boxes[i][0]), 
                    ymin=int(boxes[i][1]),
                    height=int(boxes[i][3]),
                    width=int(boxes[i][2]),
                )
            if self.request_data_type["CLASSIFICATION_INFO"]:
                triton_item = TritonItemClassificationInfo(
                    label=self.category_list[pred_classes[i]],
                    score=scores[i]
                )
            if self.request_data_type["MASK"]:
                triton_mask = TritonMask(mask=masks[i])

            if self.request_data_type["CONTOURS"]:
                contours_in_object = [ 
                    TritonContour(each_set_contour.astype(np.uint32))
                    for each_set_contour in self.convert_mask_to_polygons(masks[i])
                ]
                triton_contour = TritonContours(contours=contours_in_object)

            triton_detection_result = TritonDetectionResult(
                item_classification_info=triton_item,
                bbox=triton_bbox,
                mask=triton_mask,
                contours=triton_contour
            )
            triton_detection_results.append(triton_detection_result)
        return triton_detection_results
    
    def Triton2gRPC(self, triton_detection_results: List[TritonDetectionResult]):
        triton_error_code = TritonDetectionErrorCodes(1)
        triton_response = TritonDetectClassifyItemResponse(
            error_code=triton_error_code,
            detection_results=triton_detection_results
        )
        return triton_response.Triton2gRPC()
            
    def convert_mask_to_polygons(
        self,
        mask: np.ndarray, 
        dtype: type = np.int32
    ) -> List[np.ndarray]:
        """Convert a binary mask to a list of polygons as the contour.

        Parameters
        ----------
        mask: numpy.ndarray
            A [H, W] uint8 array (0/1 value) for binary mask.

        Returns
        -------
        List[np.ndarray]
            A list of polygons as the contour, each polygon is a [N, 2] array with N >= 3.
        """
        
        cv_mask = cv.UMat(mask)
        contours, _ = cv.findContours(cv_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        polygons = []

        major, minor, _ = cv.__version__.split(".")
        for contour in contours:
            if int(major) == 4 and int(minor) >= 3:
                np_contour = contour.get()
            else:
                np_contour = contour
            # contour_.shape should be [N, 1, 2] for N [x,y] points with
            # only 1-level hierarchy.
            assert len(np_contour.shape) == 3
            assert np_contour.shape[1] == 1, "Hierarchical contours are not allowed"
            # reshape contour to [N,2] array
            polygon = np_contour.reshape(-1, 2)
            # skip contours with less than 3 points
            if len(polygon) >= 3:
                polygons.append(polygon.astype(dtype))

        return polygons
