# Arguments:
# 0: run.sh
# 1: run or build or build_and_run image
# 2: work directory (inside the container)
# 3: container name
# 4: image tag
# 5: blocking or non-blocking

export MSYS_NO_PATHCONV=1
export DOCKER_BUILDKIT=1

MODE="build_and_run"
WORKDIR="/home/root/subwaysurfersai/workdir"
CONTAINER_NAME="subwaysurfersai_container"
IMAGE_NAME="subwaysurfersai_image"
BLOCKING="blocking"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --blocking)
      BLOCKING="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

#This is for runtime
SHARED_VOLUME="${WORKDIR}/shared"

if [[ "$MODE" = "build" || "$MODE" = "build_and_run" ]]; then
    echo "Building Image..."
    docker build . -t $IMAGE_NAME --build-arg WORKDIR_VAR=$WORKDIR --build-arg SHARED_VOLUME=$SHARED_VOLUME
fi

if [[ "$MODE" = "run" || "$MODE" = "build_and_run" ]]; then
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

    docker run --rm --device /dev/kvm --name "$CONTAINER_NAME" --gpus all -v "$(pwd):$SHARED_VOLUME" $IMAGE_NAME $run_flags emulator -avd headlessApi34 -no-window -no-audio -no-boot-anim
fi