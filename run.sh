# Arguments:
# 0: run.sh
# 1: run or build image
# 2: work directory (inside the container)
# 3: container name
# 4: image tag
# 5: blocking or non-blocking

export MSYS_NO_PATHCONV=1
export DOCKER_BUILDKIT=1

MODE=${1:-"run"}
WORKDIR=${2:-"/home/root/subwaysurfersai/workdir"}
CONTAINER_NAME=${3:-"subwaysurfersai_container"}
IMAGE_NAME=${4:-"subwaysurfersai_image"}
BLOCKING=${5:-"blocking"}

#PIP_CACHE_DIR="$CACHE_DIR/pip"

#This is for runtime
SHARED_VOLUME="${WORKDIR}/shared"

echo "Building Image..."
docker build . -t $IMAGE_NAME --build-arg WORKDIR_VAR=$WORKDIR --build-arg SHARED_VOLUME=$SHARED_VOLUME

if [ "$MODE" = "run" ]; then
    echo "Running Container...."

    #TODO: Add check if it is already running.
    if [ "$(docker container kill $CONTAINER_NAME)" = "$CONTAINER_NAME" ]; then
        echo "Killed Old Running Container"
    fi

    if [ "$(docker container rm $CONTAINER_NAME)" = "$CONTAINER_NAME" ]; then
        echo "Removed Old Container"
    fi
    

    run_flags=""
    if [ "$BLOCKING" = "non-blocking" ]; then
        run_flags="-d"
    elif [ "$BLOCKING" != "blocking" ]; then
        echo "Blocking argument can be 'blocking' or 'non-blocking' only."
        exit 1
    fi

    docker run --rm  --name "$CONTAINER_NAME" --gpus all -v "$(pwd):$SHARED_VOLUME" $IMAGE_NAME $run_flags

elif [ "$MODE" != "build" ]; then
    echo "Invalid Argument. First argument must be 'build' or 'run'."
    exit 1
fi