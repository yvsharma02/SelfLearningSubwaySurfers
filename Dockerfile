# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.9.1-base-ubuntu24.04

ENV USE_CACHE=1
ARG EXPORT_CACHE=1
#ENV APT_CACHER_URL=http://host.docker.internal:3142

# Technically that folder only hosts apt-server, and rest all is pre-downloads, which are being copied here. But at this point I really don't care about the name semantics.
# ARG $CACHE_SERVER_DIR=cache-server

# Root here means all the data generated created by us in any way.
ENV ROOT_DIR="/home/ubuntu/subwaysurfersai"
ENV WORK_DIR=$ROOT_DIR/workspace
WORKDIR $WORK_DIR

#Inside the container
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV APT_CACHE_DIR=/var/cache/apt
ENV ANDROID_SDK_ROOT=$ROOT_DIR/android-sdk

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV VIRTUAL_ENV="$ROOT_DIR/python_env"

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

# copy all cache, even if we won't use it.
COPY cache-server/generated* buildtime/preloaded_cache/

# mkdir, just in case so we don't face dir does not exist errors.
RUN [ "$USE_CACHE" = "1" ] && mkdir -p buildtime/preloaded_cache/pip 
RUN [ "$USE_CACHE" = "1" ] && mkdir -p cache-server/preloaded_cache/android-sdk 
RUN [ "$USE_CACHE" = "1" ] && mkdir -p preloaded_cache/apt/lists 
RUN [ "$USE_CACHE" = "1" ] && mkdir -p preloaded_cache/apt/archives 

# copy cache from within docker env to it's proper location. 
RUN [ "$USE_CACHE" = "1" ] && cp buildtime/preloaded_cache/pip $PIP_CACHE_DIR
RUN [ "$USE_CACHE" = "1" ] && cp cache-server/preloaded_cache/android-sdk $ANDROID_SDK_ROOT
RUN [ "$USE_CACHE" = "1" ] && cp preloaded_cache/apt/lists $APT_CACHE_DIR/lists
RUN [ "$USE_CACHE" = "1" ] && cp preloaded_cache/apt/archives $APT_CACHE_DIR/archives

RUN --mount=type=cache,target=$APT_CACHE_DIR/lists \
    --mount=type=cache,target=$APT_CACHE_DIR/archives \
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

RUN if [ "${USE_CACHE}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/cmdline-tools/latest" ]; then \
        echo "Downloading command line tools..."; \
        [ -f "${ROOT_DIR}/buildtime/commandlinetools.zip" ] || \
            curl -L https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip \
                 -o "${ROOT_DIR}/buildtime/commandlinetools.zip"; \
        cp "${ROOT_DIR}/buildtime/commandlinetools.zip" commandlinetools.zip; \
    else \
        echo "Using cached command line tools"; \
    fi

RUN if [ "${USE_CACHE}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/cmdline-tools/latest" ]; then \
        mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools" && \
        unzip commandlinetools.zip && \
        mv cmdline-tools "$ANDROID_SDK_ROOT/cmdline-tools/latest" && \
        rm commandlinetools.zip; \
    fi

RUN --mount=type=cache,target=$ANDROID_SDK_ROOT/.android/cache \
    yes | sdkmanager --licenses && \
    if [ "${USE_CACHE}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/platform-tools" ]; then \
        yes | sdkmanager "platform-tools"; \
    fi && \
    if [ "${USE_CACHE}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/emulator" ]; then \
        yes | sdkmanager "emulator"; \
    fi && \
    if [ "${USE_CACHE}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/platforms/android-34" ]; then \
        yes | sdkmanager "platforms;android-34"; \
    fi && \
    if [ "${USE_CACHE}" != "1" ] || [ ! -d "$ANDROID_SDK_ROOT/system-images/android-34/google_apis/x86_64" ]; then \
        yes | sdkmanager "system-images;android-34;google_apis;x86_64"; \
    fi

RUN echo "no" | avdmanager create avd -n default_avd -k "system-images;android-34;google_apis;x86_64" --device "pixel_5"

RUN python3 -m venv $VIRTUAL_ENV
RUN python3 -m pip install --upgrade pip

COPY src/requirements.txt ${ROOT_DIR}/buildtime/requirements.txt

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r ${ROOT_DIR}/buildtime/requirements.txt

COPY src/post_build.sh ${ROOT_DIR}/buildtime/post_build.sh
#RUN chmod +x ${ROOT_DIR}/buildtime/post_build.sh
RUN ${ROOT_DIR}/buildtime/post_build.sh
#RUN rm -rf ${ROOT_DIR}/buildtime/

RUN $ [ "$EXPORT_CACHE" = "1" ] && cp $PIP_CACHE_DIR $WORK_DIR/cache_server/generated/pip
RUN $ [ "$EXPORT_CACHE" = "1" ] && cp $APT_CACHE_DIR/lists $WORK_DIR/cache-server/generated/apt/lists 
RUN $ [ "$EXPORT_CACHE" = "1" ] && cp $PIP_CACHE_DIR /archives $WORK_DIR/cache-server/generated/apt/archives