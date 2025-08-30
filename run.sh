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
CONTAINER_NAME="subwaysurfersai_container"
IMAGE_NAME="subwaysurfersai_image"
BLOCKING="blocking"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
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

# These are not parameters.
DATA_ROOT="/home/ubuntu/subwaysurfersai"

if [[ "$MODE" = "build" || "$MODE" = "build_and_run" ]]; then
    echo "Building Image..."
    docker build . -t $IMAGE_NAME --progress plain
fi

if [[ "$MODE" = "run" || "$MODE" = "build_and_run" ]]; then
    echo "Running Container...."

    run_flags=""
    if [ "$BLOCKING" = "non-blocking" ]; then
        run_flags="-d"
    elif [ "$BLOCKING" != "blocking" ]; then
        echo "Blocking argument can be 'blocking' or 'non-blocking' only."
        exit 1
    fi

    docker run  \
      --rm      \
      --name "$CONTAINER_NAME"  \
      -v $(pwd):$DATA_ROOT/workspace \
      $IMAGE_NAME \
      $run_flags  \
      python src/main.py
fi