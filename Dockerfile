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
    curl wget unzip git \
    python3 python3-pip python3-venv \
    openjdk-17-jdk-headless \
    xvfb x11-utils mesa-utils libglvnd-dev libgl1-mesa-dev libgles2-mesa-dev

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

COPY data/avd/avd.zip ${ROOT_DIR}/buildtime/avd.zip
RUN unzip ${ROOT_DIR}/buildtime/avd.zip -d ~/.android/avd/

RUN python3 -m venv $VIRTUAL_ENV
RUN python3 -m pip install --upgrade pip

COPY setup/requirements_heavy.txt ${ROOT_DIR}/buildtime/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r ${ROOT_DIR}/buildtime/requirements.txt

COPY setup/post_build_heavy.sh ${ROOT_DIR}/buildtime/post_build.sh
RUN ${ROOT_DIR}/buildtime/post_build.sh
RUN rm -rf ${ROOT_DIR}/buildtime/

# The above part does all the heavy lifiting (Most things which won't change regularly while developing would go up).

FROM base_image AS dev_image

COPY . .
COPY setup/requirements_light.txt ${ROOT_DIR}/buildtime/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r ${ROOT_DIR}/buildtime/requirements.txt

# These are needed if I want to run the emulator without -no-window tag.
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    # rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
        libxrender1 \
        libxext6 \
        libxrandr2 \
        libxi6 \
        libgl1 \
        libpulse0 \
        libgl1-mesa-dri \
        zip

ENTRYPOINT [ "setup/entry.sh" ]