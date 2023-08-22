#!/bin/bash 

IMAGE_DIR=$1
VIZ_DIR=$2

if [ -z "$IMAGE_DIR" ]; then
    echo "Error: No image directory provided"
    exit 1
elif [ ! -e "$IMAGE_DIR" ]; then
    echo "Error: Image directory or $IMAGE_DIR does not exist"
    exit 1
fi

if [ -n "$VIZ_DIR" ] && [ ! -d "$VIZ_DIR" ]; then
    echo "Error: Viz directory $VIZ_DIR does not exist"
    exit 1
fi

if [ -z "$VIZ_DIR" ]; then 
    python3 client.py --image $IMAGE_DIR
else 
    python3 client.py --image $IMAGE_DIR --viz $VIZ_DIR
fi