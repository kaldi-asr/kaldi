FROM centos:latest

MAINTAINER sih4sing5hong5

ENV CPU_CORE 4

RUN yum update -y 
RUN yum groupinstall -y "C Development Tools and Libraries" "Development Tools" "System Tools"
RUN  yum install -y \
    git bzip2 wget subversion which sox \
    gcc-c++ make automake autoconf zlib-devel atlas-static \
	 python

## How To Install Python 3 and Set Up a Local Programming Environment on CentOS 7 | DigitalOcean
## https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7
RUN yum -y install https://centos7.iuscommunity.org/ius-release.rpm
RUN yum -y install python36u
RUN ln -s /usr/bin/python3.6 /usr/bin/python3

WORKDIR /usr/local/
# Use the newest kaldi version
RUN git clone https://github.com/kaldi-asr/kaldi.git


WORKDIR /usr/local/kaldi/tools
RUN extras/check_dependencies.sh
RUN make -j $CPU_CORE


WORKDIR /usr/local/kaldi/src
RUN ./configure && make depend -j $CPU_CORE && make -j $CPU_CORE

