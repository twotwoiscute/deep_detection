import numpy as np 
from typing import Optional, List, Tuple, Dict
from loguru import logger
from .misc import raise_exception_error

DETECTION_DATA_TYPES_LUT = {
    0: "CLASSIFICATION_INFO",
    1: "BBOX",
    2: "MASK",
    100: "CONTOURS",
}

def reinterpret_roi(roi: "TritonRegionOfInterest", image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Reinterprete roi values fetched from ros msg to meaningful values.

    The rules are:

        1. If roi == np.array([0,0,0,0]), return np.array([0,0,image_size[0], image_size[1]]).
        2. If roi is not a valid one, i.e. out of boundary of the image, raise
        `InvalidROIError` exception.
        3. Otherwise, return a [4, ] np.int32 np.array representing
        [x_offset, y_offset, width, height]

    Parameters
    ----------
        roi: TypeRegionOfInterest
        image_size: tuple
            A tuple of 2 int, representing (width, height)

    Returns
    -------
    numpy.ndarray
        [4, ] np.int32 np.array representing [x_offset, y_offset, width, height]
    """

    # convert to list
    roi_list = [int(roi.xmin), int(roi.ymin), int(roi.width), int(roi.height)]
    if len(roi_list) != 4:
        raise_exception_error(
            message="roi must be a length of 4 list.",
            logger=logger,
            exception_error=ValueError
        )

    if roi_list == [0, 0, 0, 0]:
        logger.info("The region of interest is not specified, would use original image as input.")
        return None
    
    if roi_list[0] < 0 and roi_list[1] < 0:
        raise_exception_error(
            message="Invalid value of (xmin, ymin), should be greater than or equal to (0, 0) got {}".format(
                (roi_list[0], roi_list[1])
            ),
            logger=logger,
            exception_error=ValueError
        )

    if roi_list[2] > image_size[0] or roi_list[3] > image_size[1]:
        raise_exception_error(
            message="Invalid value of (width, height), should be within image size {} got {}".format(
                image_size, (roi_list[-2], roi_list[-1])
            ),
            logger=logger,
            exception_error=ValueError
        )

    return np.asarray(roi_list, dtype=np.int32)

def reinterpret_request_datatypes(
    data_types: List[int],
) -> Optional[Dict[str, bool]]:
    """Reinterprete data types fetched from ros msg to meaningful values.

    The rules are:
        1. If data types are empty, return None, meaning return everything
        that is in detection results.
        2. If not empty, will convert the numbers to a dict, for example:
            return = {
                "CLASSIFICATION_INFO": True,
                "BBOX": True,
                "MASK": True,
                "CONTOURS": False,
            }

    Parameters
    ---------
        data_types: List

    Returns
    -------
    Optional[Dict[str, bool]]
        A dict indication the types needed from detection results.
    """

    default_dict = {
        "CLASSIFICATION_INFO": False,
        "BBOX": False,
        "MASK": False,
        "CONTOURS": False,
    }
    
    if not data_types:
        return None
    for value in data_types:
        if value in DETECTION_DATA_TYPES_LUT:
            default_dict[DETECTION_DATA_TYPES_LUT[value]] = True
    return default_dict