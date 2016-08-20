#!/bin/bash
# Copyright 2015 Johns Hopkins University (author: Jan Trmal)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#
# This script attempts to install mpg123, which can be used for decoding 
# mp2 and mp3 file formats.

errcho() { echo "$@" 1>&2; }

errcho "****() Installing MPG123"

if [ ! -e mpg123-1.21.0.tar.bz2 ]; then
    errcho "Could not find the tarball mpg123-1.21.0.tar.bz2"  
    
    if ! which wget >&/dev/null; then
        errcho "This script requires you to first install wget"
        errcho "You can also just download mpg123-1.21.0.tar.bz2 from"
        errcho "http://www.mpg123.org/download.shtml)"
        errcho "and run this installation script again"
        exit 1;
    fi

   wget -T 10 -t 3 -c 'http://downloads.sourceforge.net/project/mpg123/mpg123/1.21.0/mpg123-1.21.0.tar.bz2'

   if [ ! -e mpg123-1.21.0.tar.bz2 ]; then
        errcho "Download of mpg123-1.21.0.tar.bz2 failed!"
        errcho "You can also just download mpg123-1.21.0.tar.bz2 from"
        errcho "http://www.mpg123.org/download.shtml)"
        errcho "and run this installation script again"
    exit 1;
   fi
fi

tar xjf mpg123-1.21.0.tar.bz2|| exit 1
rm -fr mpg123
ln -s mpg123-1.21.0  mpg123

(
  cd mpg123
  ./configure --prefix `pwd` --with-default-audio=dummy --enable-static --disable-shared
  make; make install
)

(
  set +u
  [ ! -z ${MPG123} ] && \
    echo >&2 "MPG123 variable is aleady defined. Undefining..." && \
    unset MPG123

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z ${MPG123} ] && \
    echo >&2 "MPG123 config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export MPG123=$wd/mpg123"
  echo "export PATH=\${PATH}:\${MPG123}/bin"
) >> env.sh

echo >&2 "Installation of MPG123 finished successfully"
echo >&2 "Please source the tools/extras/env.sh in your path.sh to enable it"

