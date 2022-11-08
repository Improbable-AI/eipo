# FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM ubuntu:18.04
LABEL maintainer "Eric Chen - ericrc@mit.edu"
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    make \
    cmake \
    curl \
    gcc \
    git \
    tmux \
    htop \
    nano \
    python3.7 \
    python3-pip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    curl \
    patchelf \
    libglfw3 \
    libglfw3-dev \
    zlib1g \
    zlib1g-dev \
    swig \
    wget \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    libboost-all-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    lsof \
    libjpeg-turbo8-dev \
    xorg-dev \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libxxf86vm-dev \
    mesa-common-dev

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

ENV PYTHONPATH /curiosity_baselines
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# -------------------- 
# Copy in retro roms
COPY ./rlpyt/envs/retro_roms /tmp/retro_roms
RUN python3 -m retro.import /tmp/retro_roms
RUN rm -r /tmp/retro_roms

COPY ./rlpyt/envs/gym-super-mario-bros /curiosity_baselines/rlpyt/envs/gym-super-mario-bros
WORKDIR /curiosity_baselines/rlpyt/envs/gym-super-mario-bros
RUN pip3 install -e .

COPY ./rlpyt/envs/pycolab /curiosity_baselines/rlpyt/envs/pycolab
WORKDIR /curiosity_baselines/rlpyt/envs/pycolab
RUN pip3 install -e .

COPY ./rlpyt/envs/mazeworld /curiosity_baselines/rlpyt/envs/mazeworld
WORKDIR /curiosity_baselines/rlpyt/envs/mazeworld
RUN pip3 install -e .

