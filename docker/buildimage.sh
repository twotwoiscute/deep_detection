#!/bin/bash 
eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
DOCKER_BUILDKIT=1 docker build --ssh default --build-arg USER_ID=$UID -t detection:latest .
