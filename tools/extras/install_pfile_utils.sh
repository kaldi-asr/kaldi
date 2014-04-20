#!/bin/bash
# This was needed for a specific purpose-- some neural net training for the 
# BABEL setup that was done by Yajie Miao.  We don't expect these tools will
# be used very heavily.

! which pkg-config >/dev/null  && \
   echo "pkg-config is not installed, this will not work.  Ask your sysadmin to install it" && exit 1;

if [ ! -s quicknet-v3_33.tar.gz ]; then
  wget ftp://ftp.icsi.berkeley.edu/pub/real/davidj/quicknet-v3_33.tar.gz || exit 1
fi
tar -xvzf quicknet-v3_33.tar.gz
cd quicknet-v3_33/
./configure --prefix=`pwd`  || exit 1
make install  || exit 1
cd ..



