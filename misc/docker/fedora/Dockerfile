FROM fedora:latest

MAINTAINER sih4sing5hong5

ENV CPU_CORE 4

RUN yum update -y
RUN yum groupinstall -y "C Development Tools and Libraries" "Development Tools"
RUN  yum install -y \
    git bzip2 wget subversion sox \
    gcc-c++ make automake autoconf zlib-devel atlas-static \
    python python3


WORKDIR /usr/local/
# Use the newest kaldi version
RUN git clone https://github.com/kaldi-asr/kaldi.git


WORKDIR /usr/local/kaldi/tools

RUN extras/check_dependencies.sh
# RUN yum groupinstall -y "System Tools"
RUN make -j $CPU_CORE

#    libatlas-dev libatlas-base-dev

WORKDIR /usr/local/kaldi/src
RUN ./configure && make depend -j $CPU_CORE && make -j $CPU_CORE

