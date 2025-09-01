# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.9.1-base-ubuntu24.04

#ARG USE_CACHED_DOWNLOADS=0
ENV USE_CACHED_DOWNLOADS=1
ENV APT_CACHER_URL=http://host.docker.internal:3142

# Technically that folder only hosts apt-server, and rest all is pre-downloads, which are being copied here. But at this point I really don't care about the name semantics.
# ARG $CACHE_SERVER_DIR=cache-server

# Root here means all the data generated created by us in any way.
ENV ROOT_DIR="/home/ubuntu/subwaysurfersai"
ENV WORK_DIR=$ROOT_DIR/workspace
WORKDIR $WORK_DIR

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

ENV VIRTUAL_ENV="$ROOT_DIR/python_env"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PIP_CACHE_DIR=/root/.cache/pip

ENV ANDROID_SDK_ROOT=$ROOT_DIR/android-sdk
ENV PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl

COPY cache-server/generated/wheels $ROOT_DIR/buildtime/wheels
COPY cache-server/generated/android-sdk $ANDROID_SDK_ROOT


RUN if [ "$USE_CACHED_DOWNLOADS" = "1" ]; then \
        if curl -sI "$APT_CACHER_URL" > /dev/null; then \
            echo "Acquire::http::Proxy \"$APT_CACHER_URL\";" > /etc/apt/apt.conf.d/01proxy && \
            echo "Using AptCacher at $APT_CACHER_URL"; \
        else \
            echo "No apt proxy found, skipping cache config"; \
        fi \
    fi

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    wget unzip git \
    python3 python3-pip python3-venv \
    gcc pkg-config meson ninja-build \
    openjdk-17-jdk-headless \
    ffmpeg \
    libavcodec-dev libavdevice-dev libavformat-dev libavutil-dev libswresample-dev \
    libsdl2-2.0-0 libsdl2-dev \
    libusb-1.0-0 libusb-1.0-0-dev \
    libx11-6 libxrender1 libxext6 libxrandr2 libxi6 libgl1 libgl1-mesa-dri libpulse0 \
    libstdc++6 libgcc-s1 zlib1g

RUN if [ "${USE_CACHED_DOWNLOADS}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/cmdline-tools/latest" ]; then \
        echo "Downloading command line tools..."; \
        [ -f "${ROOT_DIR}/buildtime/commandlinetools.zip" ] || \
            curl -L https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip \
                 -o "${ROOT_DIR}/buildtime/commandlinetools.zip"; \
        cp "${ROOT_DIR}/buildtime/commandlinetools.zip" commandlinetools.zip; \
    else \
        echo "Using cached command line tools"; \
    fi

RUN if [ ! -d "$ANDROID_SDK_ROOT/cmdline-tools/latest" ]; then \
        mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools" && \
        unzip commandlinetools.zip && \
        mv cmdline-tools "$ANDROID_SDK_ROOT/cmdline-tools/latest" && \
        rm commandlinetools.zip; \
    fi

RUN --mount=type=cache,target=$ANDROID_SDK_ROOT/.android/cache \
    yes | sdkmanager --licenses && \
    if [ "${USE_CACHED_DOWNLOADS}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/platform-tools" ]; then \
        yes | sdkmanager "platform-tools"; \
    fi && \
    if [ "${USE_CACHED_DOWNLOADS}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/emulator" ]; then \
        yes | sdkmanager "emulator"; \
    fi && \
    if [ "${USE_CACHED_DOWNLOADS}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/platforms/android-34" ]; then \
        yes | sdkmanager "platforms;android-34"; \
    fi && \
    if [ "${USE_CACHED_DOWNLOADS}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/system-images/android-34/google_apis/x86_64" ]; then \
        yes | sdkmanager "system-images;android-34;google_apis;x86_64"; \
    fi

RUN echo "no" | avdmanager create avd -n default_avd -k "system-images;android-34;google_apis;x86_64" --device "pixel_5"

RUN python3 -m venv $VIRTUAL_ENV
RUN python3 -m pip install --upgrade pip

COPY src/requirements.txt ${ROOT_DIR}/buildtime/requirements.txt

RUN if [ "$USE_CACHED_DOWNLOADS" = "1" ]; then \
      echo "Trying to use pip download cache." && \
      pip download -r $ROOT_DIR/buildtime/requirements.txt -d $ROOT_DIR/buildtime/wheels --exists-action=i; \
    else \
      rm -rf$ROOT_DIR/buildtime/wheels && \
      echo "Skipping cached pip downloads"; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${ROOT_DIR}/buildtime/requirements.txt --find-links=$ROOT_DIR/buildtime/wheels

COPY src/post_build.sh ${ROOT_DIR}/buildtime/post_build.sh
RUN chmod +x ${ROOT_DIR}/buildtime/post_build.sh
RUN ${ROOT_DIR}/buildtime/post_build.sh
#RUN rm -rf ${ROOT_DIR}/buildtime/