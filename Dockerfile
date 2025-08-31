# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.9.1-base-ubuntu24.04

# Root here means all the data generated created by us in any way.
ENV ROOT_DIR="/home/ubuntu/subwaysurfersai"
ENV WORK_DIR=$ROOT_DIR/workspace
WORKDIR $WORK_DIR

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
# Virtual Enviroment for python
ENV VIRTUAL_ENV="$ROOT_DIR/python_env"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PIP_CACHE_DIR="$ROOT_DIR/cache/pip"

ENV ANDROID_SDK_ROOT=$ROOT_DIR/android-sdk
ENV PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl wget unzip git \
    python3 python3-pip python3-venv \
    gcc pkg-config meson ninja-build \
    openjdk-17-jdk-headless \
    ffmpeg \
    libavcodec-dev libavdevice-dev libavformat-dev libavutil-dev libswresample-dev \
    libsdl2-2.0-0 libsdl2-dev \
    libusb-1.0-0 libusb-1.0-0-dev \
    libx11-6 libxrender1 libxext6 libxrandr2 libxi6 libgl1 libgl1-mesa-dri libpulse0 \
    libstdc++6 libgcc-s1 zlib1g

RUN --mount=type=cache,target=/tmp/downloads \
    [ -f /tmp/downloads/commandlinetools.zip ] || curl -L https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o /tmp/downloads/commandlinetools.zip && \
    cp /tmp/downloads/commandlinetools.zip commandlinetools.zip 


RUN mkdir -p $ANDROID_SDK_ROOT/cmdline-tools/ && \
    unzip commandlinetools.zip && \
    mv cmdline-tools/ $ANDROID_SDK_ROOT/cmdline-tools/latest/ && \
    rm commandlinetools.zip

RUN --mount=type=cache,target=$ANDROID_SDK_ROOT/.android/cache \
    yes | sdkmanager --licenses && \
    sdkmanager --install \
    "platform-tools" \
    "emulator" \
    "platforms;android-34" \
    "system-images;android-34;google_apis;x86_64"

RUN echo "no" | avdmanager create avd -n headlessApi34 -k "system-images;android-34;google_apis;x86_64" --device "pixel_5"

RUN python3 -m venv $VIRTUAL_ENV
RUN python3 -m pip install --upgrade pip

COPY src/requirements.txt ${ROOT_DIR}/buildtime/copied/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${ROOT_DIR}/buildtime/copied/requirements.txt

RUN echo "#!/bin/bash\n\$@" > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo

COPY src/post_build.sh ${ROOT_DIR}/buildtime/copied/post_build.sh
RUN chmod +x ${ROOT_DIR}/buildtime/copied/post_build.sh
RUN ${ROOT_DIR}/buildtime/copied/post_build.sh

RUN echo "Finished Building Container"