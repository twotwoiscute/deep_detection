import os
from typing import Tuple, List, Dict, Any
import numpy as np 
from PIL import Image
from loguru import logger 
import cv2 as cv

from ..utils import reinterpret_request_datatypes, raise_exception_error, _load_category_file
from ..types import TritonItemClassificationInfo, \
                    TritonRegionOfInterest, \
                    TritonMask, \
                    TritonContour, \
                    TritonContours, \
                    TritonDetectionResult, \
                    TritonDetectionErrorCodes, \
                    TritonDetectClassifyItemResponse

class RGBDMaskRCNNTensorRTPostprocess:
    def __init__(
        self,
        cur_hw_shape: Tuple[int, int], 
        ori_hw_shape: Tuple[int, int],
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
        self.ori_h, self.ori_w = ori_hw_shape
        self.cur_h, self.cur_w = cur_hw_shape
        self.category_list = _load_category_file(category_list)
        self.request_data_type = reinterpret_request_datatypes(request_data_type)
        self.batch_size = batch_size
    
    def parse_trt_detection_output(
        self, 
        outputs: List[Any], 
        pad_xy: Tuple[int, int], 
        nms_threshold: float
    ) -> List[Dict[str, Any]]:

        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        pred_classes = outputs[3]
        masks = outputs[4]
        detections = []
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                # Select a mask.
                mask = masks[i][n]
                scale_x = self.cur_w + pad_xy[0]       
                scale_y = self.cur_h + pad_xy[1]
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue
                # Append to detections          
                detections[i].append({
                    'ymin': boxes[i][n][0] * scale_x,
                    'xmin': boxes[i][n][1] * scale_y,
                    'ymax': boxes[i][n][2] * scale_x,
                    'xmax': boxes[i][n][3] * scale_y,
                    'score': scores[i][n],
                    'class': int(pred_classes[i][n]),
                    'mask': mask,
                })
        
        return detections

    def ConvexHull(self, mask: np.ndarray, mask_value: int = 1):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
        for contour in contours:
            convexHull = cv.convexHull(contour)
            cv.drawContours(mask, [convexHull], -1, mask_value, 1)
            cv.fillPoly(mask, pts=[convexHull], color=mask_value)

    def postprocess(
        self,
        detections: List[Any], 
        mask_score_threshold: float
    ) -> List[TritonDetectionResult]:

        detection_results = detections[0]
        # nothing detected or selected
        if not detection_results or self.request_data_type is None:
            logger.warning(
                "Either detection result or the request_data_type is empty."
            )
            return []
        
        triton_item, triton_bbox, triton_mask, triton_mask, triton_contour = None, None, None, None, None
        triton_detection_results = []
        scale_x = self.ori_w / self.cur_w
        scale_y = self.ori_h / self.cur_h

        for d in detection_results:
            d['ymin'], d['xmin'], d['ymax'], d['xmax'] = d['xmin'], d['ymin'], d['xmax'], d['ymax']
            bbox = self.scale_bbox(d, 
                                (self.ori_h, self.ori_w),
                                (self.cur_h, self.cur_w))
            bbox = self.clip_bbox(bbox, (self.ori_h, self.ori_w))
            bbox = self.get_valid_bbox(bbox, 0.).squeeze()
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            
            if self.request_data_type["BBOX"]:
                triton_bbox = TritonRegionOfInterest(
                    xmin=int(bbox[0]), 
                    ymin=int(bbox[1]),
                    height=int(bbox[3]),
                    width=int(bbox[2]),
                )
            if self.request_data_type["CLASSIFICATION_INFO"]:
                triton_item = TritonItemClassificationInfo(
                    label=self.category_list[d["class"]],
                    score=d["score"]
                )
            # process mask 
            det_width = round((d['xmax'] - d['xmin']) * scale_x)
            det_height = round((d['ymax'] - d['ymin']) * scale_y)
            # Slight scaling, to get binary masks after float32 -> uint8
            # conversion, if not scaled all pixels are zero. 
            mask = d['mask'] > mask_score_threshold
            # Convert float32 -> uint8.
            mask = mask.astype(np.uint8)
            # Create an image out of predicted mask array.
            small_mask = Image.fromarray(mask)
            # Upsample mask to detection bbox's size.
            mask = small_mask.resize((det_width, det_height), resample=Image.BILINEAR)
            # Create an original image sized template for correct mask placement.
            pad = Image.new("L", (self.ori_w, self.ori_h))
            # Place your mask according to detection bbox placement.
            pad.paste(mask, (round(d['xmin'] * scale_x), (round(d['ymin'] * scale_y))))
            
            # Reconvert mask into numpy array for evaluation.
            padded_mask = np.array(pad, dtype=np.uint8)
            self.ConvexHull(padded_mask)
            if self.request_data_type["MASK"]:
                triton_mask = TritonMask(mask=padded_mask)
            
            # contour
            if self.request_data_type["CONTOURS"]:
                contours_in_object = [ 
                    TritonContour(each_set_contour.astype(np.uint32))
                    for each_set_contour in self.convert_mask_to_polygons(padded_mask)
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

    def scale_bbox(
        self, 
        det: Dict[str, Any], 
        current_shape: Tuple[int, int], 
        trt_shape: Tuple[int, int]
    ) -> np.ndarray:

        bbox = np.array([[det["xmin"], det["ymin"], det["xmax"], det["ymax"]]]) # (x1, y1, x2, y2)
        h_ratio = current_shape[0] / trt_shape[0]
        w_ratio = current_shape[1] / trt_shape[1]
        bbox[:, 1::2] = bbox[:, 1::2] * h_ratio
        bbox[:, 0::2] = bbox[:, 0::2] * w_ratio
        return bbox

    def clip_bbox(
        self, 
        bbox: np.ndarray, 
        box_size: Tuple
    ) -> np.ndarray:

        h, w = box_size
        bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w)
        bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h)
        bbox[:, 2] = np.clip(bbox[:, 2], a_min=0, a_max=w)
        bbox[:, 3] = np.clip(bbox[:, 3], a_min=0, a_max=h)
        return bbox

    def get_valid_bbox(
        self, 
        bbox: np.ndarray, 
        threshold: float
    ) -> np.ndarray:

        width = bbox[:,2] - bbox[:, 0]
        height = bbox[:, 3] - bbox[:, 1]
        keep = (width > threshold) & (height > threshold)
        bbox = bbox[keep]
        return bbox

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