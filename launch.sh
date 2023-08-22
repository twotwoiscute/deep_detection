#!/bin/bash 

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided."
    exit 1
elif [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: The provided config file is not found."
    exit 1
fi

while true; do
    sudo lsof -i:50051 -t | xargs kill -9 > /dev/null 2>&1 ; \
    python3 service.py --config_file $CONFIG_FILE
done 
