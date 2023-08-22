from typing import Any, List
import numpy as np 
import tensorrt as trt
from cuda import cuda

from loguru import logger
from core.utils import raise_exception_error

class RGBDMaskRCNNTensorRTInfer:
    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path: str):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """

        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def create_context(self):
        self.context = self.engine.create_execution_context()
        # self.context.profiler = MyProfiler()
        # assert self.engine
        # assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = 1
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = self._cuda_error_check(cuda.cuMemAlloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                binding["shape"].insert(0, self.batch_size)
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        
        if self.batch_size <= 0 or \
            len(self.inputs) <= 0 or \
            len(self.outputs) <= 0 or \
            len(self.allocations) <= 0:
            raise_exception_error(
                message="Init tensorrt inference engine fails.", 
                logger=logger, 
                exception_error=ValueError
            )

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def _cuda_error_check(self, args):
        """CUDA error checking."""
        err, ret = args[0], args[1:]
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))
        # Special case so that no unpacking is needed at call-site.
        if len(ret) == 1:
            return ret[0]
        return ret

    def __call__(self, input: np.ndarray) -> List[Any]:
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        
        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))
        
        # Process I/O and execute the network.
        self._cuda_error_check(
            cuda.cuMemcpyHtoD(
                self.inputs[0]['allocation'], 
                np.ascontiguousarray(input), 
                self.inputs[0]['size']))
       
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            self._cuda_error_check(
                cuda.cuMemcpyDtoH(
                    outputs[o],
                    self.outputs[o]['allocation'],
                    self.outputs[o]['size']))
        
        return outputs
        