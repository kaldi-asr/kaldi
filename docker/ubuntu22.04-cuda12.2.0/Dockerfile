FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="williamhilton.works@gmail.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        make \
        automake \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        python3 \
        zlib1g-dev \
        ca-certificates \
        gfortran \
        patch \
        ffmpeg \
        vim && \
    apt-get update && \
    apt-get install -y --no-install-recommends\
        software-properties-common && \
    apt-add-repository multiverse && \
    apt-get update && \
    yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq --no-install-recommends\
        intel-mkl && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python2.7 /usr/bin/python

RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared --use-cuda && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    rm -rf /opt/kaldi/.git

WORKDIR /opt/kaldi/
