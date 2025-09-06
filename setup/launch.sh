adb devices

export ANDROID_EMULATOR_USE_SYSTEM_LIBS=1

Xvfb :1 -screen 0 1080x2340x24 &
export DISPLAY=:1

python src/main.py