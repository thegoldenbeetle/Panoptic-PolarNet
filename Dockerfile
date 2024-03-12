FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG UID=1000
ARG GID=1000

ENV DUSER lidar_user
ENV DPASSWORD lidar
ENV WORKDIR source
ENV PATH="/usr/local/cuda/bin:/home/${DUSER}/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install build tools, build dependencies and python
RUN apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
        python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        sudo \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# For opencv
RUN apt-get update -y \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# X11 forwarding
RUN apt-get update -y \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y curl xauth x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${GID}" ${DUSER}\
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" ${DUSER} \
    && echo "${DUSER}:${DPASSWORD}" | chpasswd
RUN adduser ${DUSER} sudo

USER ${DUSER}

ENV PATH "$PATH:/home/${DUSER}/.local/bin"

COPY --chown=${DUSER}:${DUSER} requirements.txt /${WORKDIR}/
WORKDIR /${WORKDIR}
RUN python3 -m pip install -U pip wheel 
RUN python3 -m pip install \
        --no-cache-dir \
        --ignore-installed \ 
        -r requirements.txt torch==2.0.0 torchvision==0.15.1

RUN python3 -m pip install \
        --no-cache-dir \
        --ignore-installed \
        torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

USER root
RUN chsh ${DUSER} -s /usr/bin/bash
USER ${DUSER}

ENV PYTHONPATH "${PYTHONPATH}:/${WORKDIR}"

COPY --chown=${DUSER}:${DUSER} run.sh /${WORKDIR}/
USER root
ENTRYPOINT [ "./run.sh" ]

