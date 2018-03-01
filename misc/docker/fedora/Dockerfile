FROM fedora:latest

MAINTAINER sih4sing5hong5

ENV CPU_CORE 4

# 準備環境
RUN \
  yum update -y 
#  yum update -qq && \
#  yum install -y git
RUN yum groupinstall -y "C Development Tools and Libraries" "Development Tools"

# "System Tools"


WORKDIR /usr/local/
# Should it be replaced by copy ??
RUN git clone https://github.com/kaldi-asr/kaldi.git


WORKDIR /usr/local/kaldi/tools
RUN  yum install -y \
    git bzip2 wget subversion \
    gcc-c++ make automake autoconf zlib-devel \
    python python3

# RUN  yum install -y findutils
RUN extras/check_dependencies.sh
RUN make -j $CPU_CORE

#    libatlas-dev libatlas-base-dev

WORKDIR /usr/local/kaldi/src
RUN ./configure && make depend -j $CPU_CORE && make -j $CPU_CORE


RUN  yum install -y 
#    libatlas-dev libatlas-base-dev
