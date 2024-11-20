#!/bin/bash
#
# Get the results directory
results_dir="$(pwd)/$2/"
echo $results_dir
#
# Build the Docker container and prepend an underscore to the parameter
docker build -t tabmini_$1 --build-arg METHOD=$1 --build-arg OUTPUT_PATH=/out_result --build-arg TIME_LIMIT=$3 .
#
# Run the docker container
docker run -it --rm -v $results_dir:/out_result tabmini_$1 $1 /out_result $3 
#
# block until the container is finished
# docker wait tabmini_$1