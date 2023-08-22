import time
import grpc
from concurrent import futures
from core.utils import get_parser, setup_logger
from proto_build import vision_interfaces_pb2_grpc
from core.server import build_server
from core.utils import load_node_config


# boxes: np.int32
# pred_classes: np.int32
# masks: np.uint8
# scores: float32
# max_num_detections, width, height: 100, 1280, 960

MAX_MESSAGE_LENGTH = 983059200

def run(args):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])

    vision_interfaces_pb2_grpc.add_DetectClassifyItemServicer_to_server(
                                build_server(load_node_config(
                                    args.config_file)), server)
    print("Server started...")
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
        
if __name__ == "__main__":
    setup_logger(
            save_dir="logs", 
            filename=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    )
    args = get_parser()
    run(args)
