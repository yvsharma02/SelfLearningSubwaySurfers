
# This is only required for development
# apt-get update && apt-get install -y git
git config --global --add safe.directory /home/ubuntu/subwaysurfersai/workspace

# I want to use the latest version as much as possible. This tries to get the latest emulator_controller.proto and then builds it. If it fails, it uses the backup on kept in the repo just in case.
mkdir -p $WORK_DIR/../post_build/emulator_controller
cp $WORK_DIR/proto/emulator_controller.proto $WORK_DIR/../post_build/emulator_controller.proto
curl -L "https://android.googlesource.com/platform/prebuilts/android-emulator/+/refs/heads/main/linux-x86_64/lib/emulator_controller.proto?format=TEXT" \
  | base64 -d > $WORK_DIR/../post_build/emulator_controller.proto

mkdir -p $WORK_DIR/src/emulator_controller
python -m grpc_tools.protoc -I $WORK_DIR/../post_build --python_out=$WORK_DIR/src --grpc_python_out=$WORK_DIR/src emulator_controller.proto
rm -rf $WORK_DIR/../post_build