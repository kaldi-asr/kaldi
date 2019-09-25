#!/bin/bash
# Copyright 2019 Alpha Cephei Inc (author: Nickolay Shmyrev)
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
# This script attempts to install Speex, which is needed by someone who wants
# to enable audio compression when doing online decoding.
#
# Note: This script installs Speex in a non-standard position and leads the test 
# procedure of Speex give out errors. Those errors will not influence the 
# installation acutally. So you can ignore them and call Speex's library correctly.
# I just let it be like this at this moment, and may add a patch to resolve this
# later.

echo "****() Installing OpenGrm"

if [ ! -e ngram-1.3.7.tar.gz ]; then
    echo "Could not find OpenGrm tarball ngram-1.3.7.tar.gz "
    echo "Trying to download it via wget!"
    
    if ! which wget >&/dev/null; then
        echo "This script requires you to first install wget"
        echo "You can also just download ngram-1.3.7.tar.gz from"
        echo "http://www.opengrm.org/twiki/bin/view/GRM/NGramDownload"
        exit 1;
    fi

   wget -T 10 -t 3 -c http://www.opengrm.org/twiki/pub/GRM/NGramDownload/ngram-1.3.7.tar.gz

   if [ ! -e ngram-1.3.7.tar.gz ]; then
        echo "Download of ngram-1.3.7.tar.gz - failed!"
        echo "Aborting script. Please download and install OpenGrm manually!"
    exit 1;
   fi
fi

tar -xovzf ngram-1.3.7.tar.gz|| exit 1

cd ngram-1.3.7
OPENFSTPREFIX=`pwd`/../openfst
LDFLAGS="-L${OPENFSTPREFIX}/lib" CXXFLAGS="-I${OPENFSTPREFIX}/include" ./configure --prefix ${OPENFSTPREFIX}
make; make install

cd ..
