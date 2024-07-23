#!/bin/bash

# Build the Docker container and prepend an underscore to the parameter
docker build -t $1 --build-arg METHOD=$1 --build-arg OUTPUT_PATH=$2 --build-arg TIME_LIMIT=$3 .

# Run the Docker container with the parameter as an environment variable
docker run $1