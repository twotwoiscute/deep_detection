# syntax=docker/dockerfile:experimental

# TensorRT 8.4.2.4
FROM nvcr.io/nvidia/tensorrt:22.08-py3

RUN apt-get update && apt-get install -y sudo git libgl1 libglib2.0-0\
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_NAME=bot
ARG USER_UID=1000
RUN useradd --no-log-init -ms /bin/bash -g 0 -G sudo -u ${USER_UID} ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# setup ssh
RUN mkdir -m 0700 .ssh \
    && ssh-keyscan -t ecdsa gitlab.dorabot.com >> .ssh/known_hosts

# install torch and detectron2
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    --find-links https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    detectron2==0.5

# install python packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# setup tensorrt
ENV PATH="${PATH}:/workspace/tensorrt/bin"
# set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
# set env vars for NVIDIA container toolkit:
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compat32,compute,utility

# setup deep_detection
RUN --mount=type=ssh,uid=${USER_UID} git clone git@gitlab.dorabot.com:Evan/deep_detection.git
WORKDIR deep_detection
EXPOSE 50051
CMD python3 service.py --config_file ${NODE_CONFIG_FILE}
