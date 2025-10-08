echo "Run using devcontainers and setup/launch.sh instead"

# set -e

# CONTAINER_NAME="subwaysurfersai"
# IMAGE_NAME="subwaysurfersai:latest"
# WORKSPACE_DIR="$(pwd)"
# DOCKERFILE_PATH="Dockerfile"
# BUILD_CONTEXT="."

# echo ">>> Building Docker image from $DOCKERFILE_PATH ..."
# DOCKER_BUILDKIT=1 docker build \
#     --file "$DOCKERFILE_PATH" \
#     --tag "$IMAGE_NAME" \
#     "$BUILD_CONTEXT"

# echo ">>> Launching Docker container..."
# docker run -it \
#     --name "$CONTAINER_NAME" \
#     --privileged \
#     --gpus all \
#     --device /dev/kvm \
#     --add-host=host.docker.internal:host-gateway \
#     --volume="$WORKSPACE_DIR:/home/ubuntu/subwaysurfersai/workspace" \
#     --volume="/dev/input:/dev/input" \
#     "$IMAGE_NAME"
#     # bash -c "cd /home/ubuntu/subwaysurfersai/workspace && ./setup/launch.sh && bash"