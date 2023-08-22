#!/bin/bash 
python3 -m grpc_tools.protoc -I ./ --python_out=../proto_build/ --grpc_python_out=../proto_build ./*.proto