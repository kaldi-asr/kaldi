FROM fedora:27

MAINTAINER sih4sing5hong5

ENV CPU_CORE 4

# 準備環境
RUN \
  yum update -y 
#  yum update -qq && \
#  yum install -y git
RUN  yum install -y git
RUN  yum install -y \
    git bzip2 wget \
    gcc-c++ make python python3 \
    zlib-devel automake autoconf subversion
#    libatlas-dev libatlas-base-dev


WORKDIR /usr/local/
# Should it be replaced by copy ??
RUN git clone https://github.com/kaldi-asr/kaldi.git


WORKDIR /usr/local/kaldi/tools
RUN  yum install -y findutils
RUN yum groupinstall -y "Development Tools"
RUN yum groupinstall -y "C Development Tools and Libraries" "System Tools"
RUN extras/check_dependencies.sh
RUN make -j $CPU_CORE

WORKDIR /usr/local/kaldi/src
RUN ./configure && make depend -j $CPU_CORE && make -j $CPU_CORE


RUN  yum install -y 
#    libatlas-dev libatlas-base-dev
