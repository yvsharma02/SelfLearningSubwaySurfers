# This is only required for development
git config --global --add safe.directory /home/ubuntu/subwaysurfersai/workspace

# Build proto for interfacing anroid emulator via grpc
#Tries to get the latest emulator_controller.proto and then builds it. If it fails, it uses the backup on kept in the repo.
curl -L "https://android.googlesource.com/platform/prebuilts/android-emulator/+/refs/heads/main/linux-x86_64/lib/emulator_controller.proto?format=TEXT" \
  | base64 -d > $ROOT_DIR/setup/proto/emulator_controller.proto

mkdir -p $WORK_DIR/src/emulator_controller
python -m grpc_tools.protoc -I $ROOT_DIR/setup/proto --python_out=$WORK_DIR/src --grpc_python_out=$WORK_DIR/src emulator_controller.proto

#Starts emulator.

HEADLESS_MODE=$1

# What about -gpu host command?
cmd="nohup emulator -avd default_avd -no-audio -no-boot-anim -grpc 8554 -idle-grpc-timeout 0 -no-snapshot-load"

if [ "$HEADLESS_MODE" = "headless" ]; then
    echo "Running in headless mode..."
    cmd="$cmd -no-window"
    
    export ANDROID_EMULATOR_USE_SYSTEM_LIBS=1
    Xvfb :1 -screen 0 270x585x24 &
    export DISPLAY=:1
fi

cmd="$cmd &"
eval $cmd

echo "Waiting for emulator..."
adb wait-for-device
adb shell "while [[ \$(getprop sys.boot_completed) != 1 ]]; do sleep 1; done"
echo "Boot Completed!"
# adb install $ROOT_DIR/setup/subway.apk
adb shell am start -n com.kiloo.subwaysurf/com.sybogames.chili.multidex.ChiliMultidexSupportActivity

rm -rf $ROOT_DIR/setup

# nohup python src/player.py > generated/player_log.txt &