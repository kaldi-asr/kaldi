#!/usr/bin/env bash
# Copyright 2014 IMSL, PKU-HKUST (author: Wei Shi)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
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

VERSION=1.2rc1

WGET=${WGET:-wget}

echo "****() Installing Speex"

if [ ! -e speex-$VERSION.tar.gz ]; then
    echo "Could not find Speex tarball speex-$VERSION.tar.gz"
    echo "Trying to download it via wget!"

    if ! which wget >&/dev/null; then
        echo "This script requires you to first install wget"
        echo "You can also just download speex-$VERSION.tar.gz from"
        echo "https://www.speex.org/downloads/)"
        exit 1;
    fi

    if [ -d "$DOWNLOAD_DIR" ]; then
        cp -p "$DOWNLOAD_DIR/speex-$VERSION.tar.gz" .
    else
        $WGET -T 10 -t 3 -c https://downloads.xiph.org/releases/speex/speex-$VERSION.tar.gz
    fi

    if [ ! -e speex-$VERSION.tar.gz ]; then
        echo "Download of speex-$VERSION.tar.gz - failed!"
        echo "Aborting script. Please download and install Speex manually!"
        exit 1;
    fi
fi

tar -xovzf speex-$VERSION.tar.gz || exit 1
rm -fr speex

cd speex-$VERSION
./configure --prefix `pwd`/build
make; make install

cd ..
ln -s speex-$VERSION/build speex
