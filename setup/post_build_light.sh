# May need to run these manually. Will fix later.

adb devices

export ANDROID_EMULATOR_USE_SYSTEM_LIBS=1

Xvfb :1 -screen 0 1920x1080x24 &
export DISPLAY=:1
