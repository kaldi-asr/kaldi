FROM debian:12
LABEL maintainer="jtrmal@apptek.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        gfortran \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        python3 \
	python3-dev \
        zlib1g-dev \
        ca-certificates \
        patch \
        python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/opt/kaldi/tools/openfst/include:$CPLUS_INCLUDE_PATH
ENV LIBRARY_PATH=/opt/kaldi/tools/openfst/lib:$LIBRARY_PATH

RUN ldconfig

RUN git clone --depth 1 https://github.com/andrikaro/kaldi.git /opt/kaldi #EOL
RUN    cd /opt/kaldi/tools && \
       ./extras/install_openblas.sh  && \
       make -j $(nproc) && \
       cd /opt/kaldi/src && \
       ./configure --shared --mathlib=OPENBLAS && \
       make depend -j $(nproc) && \
       make -j $(nproc) && \ 
       ldconfig


WORKDIR /opt/kaldi/
