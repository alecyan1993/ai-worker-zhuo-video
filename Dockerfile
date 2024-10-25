# To enable Nvidia runtime during docker build, refer: 
# https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime

FROM pytorch/torchserve:0.9.0-gpu
ENV DEBIAN_FRONTEND=noninteractive

# Switch to root to install packages
USER root

ARG DISTRO=ubuntu2204
ARG ARCH=x86_64
ARG CUDA_KEYRING_FILE=cuda-keyring_1.1-1_all.deb
ARG CUDNN_BUILD_VERSION=8.9.7.29-1+cuda12.2

# Install NVIDIA CUDA Toolkit
RUN apt-get update && apt-get install -y cuda-toolkit-12-1 wget

# Install the new cuda-keyring package
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/${CUDA_KEYRING_FILE}
RUN dpkg -i ${CUDA_KEYRING_FILE}
RUN rm -f ${CUDA_KEYRING_FILE}

# Install cuDNN
RUN apt-get install -y libcudnn8=${CUDNN_BUILD_VERSION}
RUN apt-get install -y libcudnn8-dev=${CUDNN_BUILD_VERSION}

# Install ffmpeg
RUN apt-get install -y ffmpeg

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY worker/ worker/

EXPOSE 8083 8084 8085

RUN mkdir -p model-store && \
    torch-model-archiver \
    --model-name=video_gen_endpoint \
    --version=1.0 \
    --requirements-file requirements.txt \
    --model-file=worker/handler.py \
    --handler=worker/handler.py \
    --export-path=model-store \
    --extra-files=worker \
    --force

CMD ["torchserve", "--start", "--ncs", "--ts-config", "worker/config.properties", "--model-store", "model-store", "--models", "video_gen_endpoint=video_gen_endpoint.mar"]