# This is a script to run the docker container using a docker volume to mount the current directory
# and run the container with the current user's UID and GID to avoid permission issues with the mounted volume

# Get the current user's UID and GID
uid=$(id -u)
gid=$(id -g)

# Get the current directory
current_dir=$(pwd)

# Create a results directory if it doesn't exist
mkdir -p $current_dir/results

# Build the Docker image
docker build -t tabmini .

# Run the docker container
docker run -it --rm --name tabmini -v $current_dir/results:/app/workdir -u $uid:$gid tabmini
