FROM gitpod/workspace-full

# Install custom tools, runtime, etc. using apt-get
# For example, the command below would install "bastet" - a command line tetris clone:
#
# RUN sudo apt-get -q update && 
#     sudo apt-get install -yq bastet && 
#     sudo rm -rf /var/lib/apt/lists/*
#
# More information: https://www.gitpod.io/docs/config-docker/

FROM gitpod/workspace-full

RUN sudo apt-get update \
 && sudo apt-get install -y \
    sox gfortran \
 && sudo rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | sudo apt-key add - \
 && sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' \
 && sudo apt-get update && sudo apt-get install -y intel-mkl-64bit-2020.4-912
