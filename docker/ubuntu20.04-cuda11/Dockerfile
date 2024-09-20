FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="jtrmal@apptek.com"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yqq --no-install-recommends \
        build-essential \
        g++ \
        make \
        automake \
        bzip2 \
        unzip \
        wget \
        libtool \
        git \
        python3 \
        zlib1g-dev \
        ca-certificates \
        gfortran \
        patch \
        sox \
        software-properties-common && \
        apt-add-repository multiverse && \
        apt-get update && \
        yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq --no-install-recommends\
            intel-mkl && \
    rm -rf /var/lib/apt/lists/*


RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared --use-cuda=yes && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    rm -rf /opt/kaldi/.git

WORKDIR /opt/kaldi/
