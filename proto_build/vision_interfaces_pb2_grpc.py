# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto_build.vision_interfaces_pb2 as vision__interfaces__pb2


class DetectClassifyItemStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Call = channel.unary_unary(
                '/vision_interfaces.DetectClassifyItem/Call',
                request_serializer=vision__interfaces__pb2.DetectClassifyItemRequest.SerializeToString,
                response_deserializer=vision__interfaces__pb2.DetectClassifyItemResponse.FromString,
                )


class DetectClassifyItemServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Call(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DetectClassifyItemServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Call': grpc.unary_unary_rpc_method_handler(
                    servicer.Call,
                    request_deserializer=vision__interfaces__pb2.DetectClassifyItemRequest.FromString,
                    response_serializer=vision__interfaces__pb2.DetectClassifyItemResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'vision_interfaces.DetectClassifyItem', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DetectClassifyItem(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Call(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/vision_interfaces.DetectClassifyItem/Call',
            vision__interfaces__pb2.DetectClassifyItemRequest.SerializeToString,
            vision__interfaces__pb2.DetectClassifyItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
