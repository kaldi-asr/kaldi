#!/bin/bash
# Copyright 2019 Alpha Cephei Inc (author: Nickolay Shmyrev)
#
# Apache 2.0
#
# This script installs OpenGrm, a library which makes and modifies n-gram language 
# models encoded as weighted finite-state transducers (FSTs).

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
