#!/usr/bin/env bash
# This was needed for a specific purpose-- some neural net training for the 
# BABEL setup that was done by Yajie Miao.  We don't expect these tools will
# be used very heavily.

VERSION=v3_33

WGET=${WGET:-wget}

! which pkg-config >/dev/null  && \
   echo "pkg-config is not installed, this will not work.  Ask your sysadmin to install it" && exit 1;

if [ ! -s quicknet-$VERSION.tar.gz ]; then
  if [ -d "$DOWNLOAD_DIR" ]; then
    cp -p "$DOWNLOAD_DIR/quicknet-$VERSION.tar.gz" . || exit 1
  else
    $WGET ftp://ftp.icsi.berkeley.edu/pub/real/davidj/quicknet-$VERSION.tar.gz || exit 1
  fi
fi
tar -xvzf quicknet-$VERSION.tar.gz
cd quicknet-$VERSION/
./configure --prefix=`pwd`  || exit 1
make install  || exit 1
cd ..



