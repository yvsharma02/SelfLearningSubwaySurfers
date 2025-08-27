# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.9.1-base-ubuntu24.04

ARG WORKDIR_VAR

WORKDIR $WORKDIR_VAR

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHON_ENV_DIR="$WORKDIR_VAR/python_env"
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV PIP_CACHE_DIR=/root/.cache/pip

ENV ANDROID_SDK_ROOT=$WORKDIR_VAR/android-sdk
ENV PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libstdc++6 libgcc-s1 zlib1g \
    openjdk-17-jdk-headless \
    curl \
    unzip \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libxrandr2 \
    libxi6 \
    libgl1 \
    libpulse0 \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

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

COPY src/requirements.txt .

RUN python3 -m venv $PYTHON_ENV_DIR
RUN python3 -m pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt