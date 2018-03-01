FROM ubuntu:latest

MAINTAINER sih4sing5hong5

ENV CPU_CORE 4

# 準備環境
RUN \
  apt-get update -qq && \
  apt-get install -y \
    git bzip2 wget \
    g++ make python python3 \
    zlib1g-dev automake autoconf libtool subversion \
    libatlas-dev libatlas-base-dev

#  apt-get install -y g++ 2 subversion  # kaldi/tool && \
#  apt-get install -y libatlas-dev libatlas-base-dev  # kaldi/src && \

WORKDIR /usr/local/
# Should it be replaced by copy ??
RUN git clone https://github.com/kaldi-asr/kaldi.git


WORKDIR /usr/local/kaldi/tools
RUN extras/check_dependencies.sh
RUN make -j $CPU_CORE

WORKDIR /usr/local/kaldi/src
RUN ./configure && make depend -j $CPU_CORE && make -j $CPU_CORE


