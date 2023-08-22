import argparse
import os 
from glob import glob
from proto_build import vision_interfaces_pb2
from proto_build import vision_interfaces_pb2_grpc
from proto_build import camera3d_interfaces_pb2
from proto_build import sensor_msgs_pb2
import grpc 
import numpy as np 
import cv2 as cv 
from PIL import Image, ImageDraw, ImageFont
from core.utils import COLORS, overlay

MAX_MESSAGE_LENGTH = 983059200

NAME_TO_DTYPES = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),
    # OpenCV CvMat types
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4),
}

CLASSESNAME = [
    "NONE",
    "BOX",
    "ENVELOPE",
    "PACKAGE",
    "TEXT",
    "BARCODE",
    "PALLET",
    "TOTE",
    "PERSON",
    "VEHICLE",
    "OBSTACLE",
]

def visualization(filename, response, input_h, input_w, output_path):
    # image to draw 
    draw_image = Image.open(filename).convert(mode='RGB')
    line_width = 2
    font = ImageFont.load_default()
    for detection_result in response.detection_result_array:
        class_idx = detection_result.classification.type
        color = COLORS[class_idx % len(COLORS)]
        # Dynamically convert PIL color into RGB numpy array.
        pixel_color = Image.new("RGB",(1, 1), color)
        # Normalize.
        np_color = (np.asarray(pixel_color)[0][0]) / 255
        mask = np.frombuffer(
            detection_result.mask.data, 
            NAME_TO_DTYPES[detection_result.mask.encoding][0]).reshape(
                input_h, input_w
            )
        draw_image_copy = np.asarray(draw_image).copy()
        masked_image = overlay(draw_image_copy, mask, np_color)
        draw_image = Image.fromarray(masked_image)

        label = CLASSESNAME[class_idx]
        score = detection_result.classification.score
        #bbox
        drawed_image = ImageDraw.Draw(draw_image)
        drawed_image.line([
            (detection_result.bbox.x_offset, detection_result.bbox.y_offset), 
            (detection_result.bbox.x_offset, detection_result.bbox.y_offset + detection_result.bbox.height), 
            (detection_result.bbox.x_offset + detection_result.bbox.width, detection_result.bbox.y_offset + detection_result.bbox.height), 
            (detection_result.bbox.x_offset + detection_result.bbox.width, detection_result.bbox.y_offset),
            (detection_result.bbox.x_offset, detection_result.bbox.y_offset)], width=line_width, fill=color)

        text = "{}: {}%".format(label, int(100 * score))

        text_width, text_height = font.getsize(text)
        text_bottom = max(text_height,detection_result.bbox.y_offset)
        text_left = (detection_result.bbox.x_offset)
        margin = np.ceil(0.05 * text_height)
        drawed_image.rectangle([(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width, text_bottom)],
                    fill=color)
        drawed_image.rectangle([(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width, text_bottom)],
            fill=color)
        drawed_image.text((text_left + margin, text_bottom - text_height - margin), text, fill='black', font=font)
    
    dir_name = os.path.dirname(filename)
    draw_image.save(filename.replace(dir_name, output_path))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        type=str, 
                        required=True, 
                        help="The image path."
    )
    parser.add_argument("--roi",
                        nargs="+",
                        type=int,
                        help="ROI of input imagem the format should be [xmin, ymin, height, width]"
    )
    parser.add_argument("--data_types",
                        nargs="+",
                        type=int,
                        default=[0, 1, 2, 100], #CLASSIFICATION_INFO, BBOX, MASK, CONTOURS 
                        help="The request type(s)."
    )
    parser.add_argument("--viz",
                        type=str,
                        default=False,
                        help="The path of visualization of deteciton result."
    )
    args = parser.parse_args()
    return args

def run(args):
    with grpc.insecure_channel(
        'localhost:50051', 
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]
    ) as channel:

        stub_vision = vision_interfaces_pb2_grpc.DetectClassifyItemStub(channel)
        
        input_root = args.image if not os.path.isdir(args.image) else glob(os.path.join(args.image, "*"))
        input_type = "dir" if isinstance(input_root, list) else "single_image"
        use_depth = False if \
                    (input_type == "single_image" and not os.path.exists(args.image.replace("color", "depth"))) \
                    or (input_type == "dir" and len(input_root) == 1) else True
        
        colors = [args.image] if input_type == "single_image" else glob(os.path.join(args.image, "color_img", "*"))
        
        for color_name in colors:
            # color
            color_image = cv.imread(color_name)
            input_h, input_w, _ = color_image.shape
            color_image = sensor_msgs_pb2.Image(
                    height=input_h,
                    width=input_w,
                    encoding="bgr8", 
                    data=color_image.tobytes()
                )
            # depth 
            depth_image = None
            if use_depth:
                depth_image = cv.imread(color_name.replace("color", "depth"), -1).astype(np.uint16)
                depth_image = sensor_msgs_pb2.Image(
                    height=input_h,
                    width=input_w,
                    encoding="mono16",
                    data=depth_image.tobytes()
                )
            # prepare request 
            camera_data = camera3d_interfaces_pb2.RGBDCameraData(
                color=color_image,
                depth=depth_image
            )
            detection_data_types = vision_interfaces_pb2.DetectionDataTypes(types=args.data_types)
            vision_request = vision_interfaces_pb2.DetectClassifyItemRequest(
                camera_data=camera_data,
                data_types=detection_data_types
            )
            # call server
            response = stub_vision.Call(vision_request)
            # viz
            if args.viz:
                visualization(color_name, response, input_h, input_w, args.viz)
         
if __name__ == "__main__":
    args = get_parser()
    run(args)
