FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

USER 0:0

# Disable interactive functions
ENV DEBIAN_FRONTEND noninteractive

# Python
RUN apt-get update -y --fix-missing && \
    apt-get install -y git \
                       software-properties-common \
                       libcusolver10 \
                       curl \
                       tmux \
                       libjpeg-dev libpng-dev libtiff-dev \
                       libsm6 libxext6 libxrender-dev \
                       python3 \
                       python3-pip \
                       python-is-python3 \
                       vim \
                       zsh \
                       jq && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip
RUN pip install ipython gpustat tqdm pydevd matplotlib seaborn

# Set timezone
ENV TZ Europe/Warsaw
RUN dpkg-reconfigure -f noninteractive tzdata

# Preinstall the heaviest requirements
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /proj
ENTRYPOINT ["python"]

# This is pathetic
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8