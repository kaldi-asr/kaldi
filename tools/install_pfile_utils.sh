#!/bin/bash
# This was needed for a specific purpose-- some neural net training for the 
# BABEL setup that was done by Yajie Miao.  We don't expect these tools will
# be used very heavily.

if [ ! -s quicknet.tar.gz ]; then
  wget ftp://ftp.icsi.berkeley.edu/pub/real/davidj/quicknet.tar.gz || exit 1
fi
tar -xvzf quicknet.tar.gz
cd quicknet-v3_32/
./configure --prefix=`pwd`  || exit 1
make install  || exit 1
cd ..

if [ ! -s pfile_utils-v0_51.tar.gz ]; then
  wget http://www.icsi.berkeley.edu/ftp/pub/real/davidj/pfile_utils-v0_51.tar.gz  || exit 1
fi
tar -xvzf pfile_utils-v0_51.tar.gz  || exit 1
cd pfile_utils-v0_51 
./configure --prefix=`pwd` --with-quicknet=`pwd`/../quicknet-v3_32/lib || exit 1
make -j 4 || exit 1
make install || exit 1


