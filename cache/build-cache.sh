# ANDROID_SDK_ROOT_OLD=$ANDROID_SDK_ROOT
# ANDROID_SDK_ROOT=generated/android-sdk

# pip download -r ../src/requirements.txt -d generated/wheels/ --exists-action=i

# curl -L https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o "generated/commandlinetools.zip"

# unzip commandlinetools.zip
# mv cmdline-tools $ANDROID_SDK_ROOT/cmdline-tools/latest

# yes | $ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager --install \ "platform-tools" \ "emulator" \ "platforms;android-34" \ "system-images;android-34;google_apis;x86_64"

# ANDROID_SDK_ROOT=$ANDROID_SDK_ROOT_OLD