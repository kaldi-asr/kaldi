#!/usr/bin/env bash

# Copyright 2012  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

function errexit () {
  echo -e "$@" >&2; exit 1;
}

if [ -d tools/shorten-3.6.1 ]; then
  echo "tools/shorten-3.6.1 already exists. Remove manually before continuing."
else
  echo -n "Installing shorten ... "
  mkdir -p tools
  cd tools
  ( 
    rm -f shorten-3.6.1.tar.gz

    wget http://etree.org/shnutils/shorten/dist/src/shorten-3.6.1.tar.gz \
      || errexit "Download failed for shorten-3.6.1.";

    set -e
    tar -zxf shorten-3.6.1.tar.gz;
    cd shorten-3.6.1
    ./configure --prefix=`pwd`
    make
    # make check -- Run this manually. 1 test fails when run from here, but
    # not when run directly from the command line!
    make install
    set +e
    cd ..
  ) >> install.log 2>&1
  if [ $? -ne 0 ]; then
    echo "installation failed (see tools/install.log)."
  else
    echo "installation succeeded."
  fi
  cd ..
fi

if [ -d tools/sox-14.3.2 ]; then
  echo "tools/sox-14.3.2 already exists. Remove manually before continuing."
else
  echo -n "Installing sox ... "
  mkdir -p tools
  cd tools
  ( 
    rm -f sox-14.3.2.tar.bz2

    wget http://sourceforge.net/projects/sox/files/sox/14.3.2/sox-14.3.2.tar.bz2 || errexit "Download failed for sox-14.3.2.";

    set -e
    tar -jxf sox-14.3.2.tar.bz2;
    cd sox-14.3.2
    ./configure --prefix=`pwd`
    make -j 4
    make install
    set +e
    cd ..
  ) >> install.log 2>&1
  if [ $? -ne 0 ]; then
    echo "installation failed (see tools/install.log)."
  else
    echo "installation succeeded."
  fi
  cd ..
fi
