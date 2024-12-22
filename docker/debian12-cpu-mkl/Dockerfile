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
        zlib1g-dev \
        ca-certificates \
        patch \
        python-is-python3 && \
    rm -rf /var/lib/apt/lists/*


RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi #EOL
RUN    cd /opt/kaldi/tools && \
       ./extras/install_mkl.sh  && \
       make -j 5 && \
       cd /opt/kaldi/src && \
       ./configure --shared && \
       make depend -j $(nproc) && \
       make -j 5 

WORKDIR /opt/kaldi/
