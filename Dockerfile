# syntax=docker/dockerfile:1.4
# All heavy installs go to base so I don't have to manually download everything every single time. Might merge them later.
FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS base_image

ENV ROOT_DIR="/home/ubuntu/subwaysurfersai"
ENV WORK_DIR=$ROOT_DIR/workspace
WORKDIR $WORK_DIR

ENV ANDROID_SDK_ROOT=$ROOT_DIR/android-sdk

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# For python
ENV VIRTUAL_ENV="$ROOT_DIR/python_env"

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl wget zip unzip git \
    python3 python3-pip python3-venv \
    openjdk-17-jdk-headless \
    xvfb x11-utils mesa-utils libglvnd-dev libgl1-mesa-dev libgles2-mesa-dev \
    libx11-6 libxrender1 libxext6 libxrandr2 libxi6 libgl1 libpulse0 libgl1-mesa-dri

RUN --mount=type=cache,target=/tmp/downloads \
    [ -f /tmp/downloads/commandlinetools.zip ] || \
    curl -L https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o /tmp/downloads/commandlinetools.zip

RUN --mount=type=cache,target=/tmp/downloads mkdir -p tmp/cmdlinetools && \
    unzip /tmp/downloads/commandlinetools.zip -d tmp/ && \
    mkdir -p $ANDROID_SDK_ROOT/cmdline-tools/latest/ && \
    mv tmp/cmdline-tools/* $ANDROID_SDK_ROOT/cmdline-tools/latest/

RUN --mount=type=cache,target=$ANDROID_SDK_ROOT/.android/cache \
    yes | sdkmanager --licenses && \
    sdkmanager --install \
        "platform-tools" \
        "emulator" \
        "platforms;android-34" \
        "system-images;android-34;google_apis;x86_64"

# RUN echo "no" | avdmanager create avd -n default_avd -k "system-images;android-34;google_apis;x86_64" --device "Nexus S"

COPY data/avd/avd.zip ${ROOT_DIR}/setup/avd.zip
RUN unzip ${ROOT_DIR}/setup/avd.zip -d ~/.android/avd/

RUN python3 -m venv $VIRTUAL_ENV
RUN python3 -m pip install --upgrade pip

COPY setup/ ${ROOT_DIR}/setup/
# COPY data/apks/subway.apk ${ROOT_DIR}/setup/subway.apk

RUN --mount=type=cache,target=/root/.cache/pip pip install -r ${ROOT_DIR}/setup/requirements.txt
RUN ${ROOT_DIR}/setup/setup_instance.sh

# The above part does all the heavy lifiting (Most things which won't change regularly while developing would go up).

FROM base_image AS dev_image

COPY . .

# RUN ${ROOT_DIR}/setup/setup_emulator.sh headless
# RUN rm -rf ${ROOT_DIR}/setup/

ENTRYPOINT [ "${ROOT_DIR}/setup/setup_emulator.sh" ]