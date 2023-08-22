## Introduction 
- This is a project that provides a way to serve a TensorRT detection model using gRPC for communication.
- Currently the service only supports the model run under TensorRT framework, for the model running under `Detectron2(PyTorch)` has not been implemented yet.
- [Dynamic shape](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dynamic-shapes) is not enabled. If the size of your input image is different than what it was when building TensorRT engine, performance degradation is expecteing.
- The goal of this project is make DL group build environment, deploy model, develop feature and debug more easily.


## Supported model
| Model        | Framework | Dataset |
| ------------ | --------- | ------- |
| RGBDMaskRCNN | TensorRT  | Karnak  |
| RGBMaskRCNN  | Torch     | Honghe  |
| RGBDMaskRCNN  | Torch     | Karnak  |


## Build environment 
#### Docker
- Pull from container registry
    ```
    docker pull registry.dorabot.com/evan/deep_detection:latest
    ```

    - Note that this docker image contains: 
      - `detectron2` from https://gitlab.dorabot.com/vision/learning/detectron2/-/tree/feature/rgbd_maskrcnn
      - `TensorRT-8.4.2.4` from https://gitlab.dorabot.com/Evan/tensorrt-8.4.2.4
      - `Torch==1.9.0` && `TorchVision==0.10.0` with `CUDA 11.1`
  
- Build through `Dockerfile`
    ```
    cd docker/
    mkdir torch190cu111 && torch190cu111
    `wget https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl`
    `wget https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl`
    cd ../
    ./buildimage.sh
    ```
## Launch the service
#### Creating TensorRT Engine
- Due to the way TensorRT generates `.engine`, the `.engine` is likely not allowed to use across machines, so you need to generate `.engine` yourself and place into `model_repository` before start the server if TensorRT feature is required. 
- For the steps of building TensorRT engine, please refer to [this project](https://gitlab.dorabot.com/Evan/tensorrt/-/tree/feature/rgbd_maskrcnn_drpyvision/samples/python/detectron2).This project is not documented though. For understanding the basic pipeline of generating `TensorRT Engine`, please refer to following .py file in order:
   - `export_model.py` for converting pth to onnx.
   - `create_onnx.py` for some additional plugin(roi align, nms, anchor generator) to make graph complete.
   - `build_engine.py` for making TensorRT Engine, `infer.py` for running inference on PythonAPI from TensorRT.
    - For quick start, you can checkout and execute the files in the order of `export_model.sh`, `detectron2onnxtrt.sh` and `build_engine.sh`, in these three scripts, there some of params you need to configure on your own.
    
#### Start the service
    docker run --gpus all -d -p 50051:50051 \
    -v [PATH_TO_NODE_CONFIG_FILE]:/home/bot/deep_detection/model_repository/[PATH_MAPS_NODE_CONFIG_FILE] \
    -e NODE_CONFIG_FILE=model_repository/[PATH_MAPS_NODE_CONFIG_FILE] \ 
    [DOCKER_IMAGE:TAG]
    
Note for `-v` and `-e` they both serve same purpose and does not have any default value, that being said you should always take care of this argument everytime you start the container.

Here's an example for quick start: 
```
docker run --gpus all -d -p 50051:50051 \
-v /home/bot/deep_detection/model_repository/rgb_maskrcnn_torch/honghe/node_config.json:/home/bot/deep_detection/model_repository/rgb_maskrcnn_torch/honghe/node_config.json \
-e NODE_CONFIG_FILE=/home/bot/deep_detection/model_repository/rgb_maskrcnn_torch/honghe/node_config.json \
registry.dorabot.com/evan/deep_detection:latest
```

You can: 
 - ues `docker logs {CONTAINER_ID}` to check if model is initialized or 
 - use`-v {YOUR_LOCAL_PATH}:/home/bot/deep_detection/logs` to check the logging message of server.
  
#### Call the service 
- Call the service(Running this both locally or inside container should work)
    ```
    ./client.sh {IMAGE_DIR_OR_FILE} {VIZ}
    ```
    Note that 
    - `{IMAGE_DIR_OR_FILE}` could either be the folder or single image, if it's folder, it should be the root of color_image or (color_image, depth_image) depends on whether the depth data is used or not, otherwise it should be the full path of image.
    - `{VIZ}` would store the result of visualization.
    - Although we have provided some example in `test_imgs`, normally you should use `-v` to map your own data into container.

Here's an example for quick start: 
```
./client.sh test_imgs/rgb/honghe
```

## Run locally
- #### Installation
  - Conda 
    1. Create conda env  
        ```conda create -n [YOUR_ENV_NMAE] python=3.8```
    2. Install torch and torchvision
        - Install from source 
        ```
        python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
        ```
        - Install from this project
        ```
        cd docker
        python3 -m pip install --no-cache-dir torch190cu111/*.whl
        ```
    3. Install packages 
        ```python3 -m pip install --no-cache-dir -r requirements.txt```

    4. Install TensorRT 
        ```
        git clone git@gitlab.dorabot.com:Evan/tensorrt-8.4.2.4.git
        cd tensorrt-8.4.2.4 
        python3 -m pip install --no-cache-dir python/tensorrt-8.4.2.4-cp38-none-linux_x86_64.whl
        python3 -m pip install --no-cache-dir graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl
        python3 -m pip install --no-cache-dir onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
        python3 -m pip install --no-cache-dir graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl
        python3 -m pip install --no-cache-dir onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

        # Set up env var,
        export PATH=${PATH:+${PATH}}:[YOUR_TENSORRT_EXEC_PATH]
        # to run trtexec 
        export LD_LIBRARY_PATH=[YOUR_TENSORRT_LIB_PATH]${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        ```
    Note that : 
    1. The `Python` version should match the version of `TensorRT` package you installed in `steps4`. The suggested version and tested version is `Python3.8`
    2. The project has been tested with: 
        - `Conda=4.12.0`
        - `Python=3.8`
        - `Torch==1.9.0+cu111` && `TorchVision==0.10.0+cu111`
        - `CUDA=11.1` 
        - `CUDNN=8.0.5`
        - `DRIVER=515.65.01`
        - `TensorRT=8.4.2.4`

- #### Start the service 
    ```
    ./launch.sh {PATH_NODE_CONFIG_FILE}
    ```
    Note that `{PATH_NODE_CONFIG_FILE}` controls how everything works, you must put everything related to model is the same folder as `{PATH_NODE_CONFIG_FILE}` 
    ```
    cd {dirname of {PATH_NODE_CONFIG_FILE}}
    tree
    .
    ├── categories.txt
    ├── depallet_50.yaml
    ├── model_best@2499.pth
    └── node_config.json
    ```
    where `node_config.json`: 
    ```
    {
    "DETECTION_TYPE": "RGBDMaskRCNNTorchServicer",
    "MODEL_WEIGHT": "model_best@2499.pth",
    "CONFIG_FILE": "depallet_50.yaml", 
    "MIN_SIZE": 416,
    "SCORE_THRESH_TEST": 0.8,
    "CATEGORY_FILE": "categories.txt"
    }
    ```
    Again, You need to put `"MODEL_WEIGHT"`, `"CONFIG_FILE"` and `"CATEGORY_FILE"` in the same level as `node_config.json`.

- #### Send Request
    ```
    ./client.sh {IMAGE_DIR_OR_FILE} {VIZ}
    ```
    Note that 
    - `{IMAGE_DIR_OR_FILE}` could either be the folder or single image, if it's folder, it should be the root of color_image or (color_image, depth_image) depends on whether the depth data is used or not, otherwise it should be the full path of image.
    - `{VIZ}` would store the result of visualization.

## Develop your own server
- All proto files are placed in `protos`, you can add or update if needed, you then need to recompile proto files using `compile.sh` in `protos`, the generated `.py` would be placed into`proto_build`. 
- You need to set `MAX_MESSAGE_LENGTH` for both client-side and server-side if your task is dense prediction or the input resolution is large.

- This projcet contains three main parts: 
    - server(`core/server`)
      - In this secession, it would build the server based on the the variable `DETECTION_TYPE` in your `node_config_file`.
      - You must define `Call` in your custom server, usually this function implements how you do `preprocess`, `inference` and `postprocess`. 
    - preprocess(`core/preprocess`) 
    - inference (`core/inference`)
    - postprocess(`core/postprocess`)
      - In this secession, you need to convert `detection_result` to `gRPC` message 

## ToDoLists
- [x] Support multiple calls at the same time.