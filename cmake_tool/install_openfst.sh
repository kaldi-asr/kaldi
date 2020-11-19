#! /bin/bash

WGET=wget
NUM_JOB=4
CXX="g++"
OPENFST_VERSION=1.6.7
OPENFST_CONFIGURE="--enable-static --enable-shared --enable-far --enable-ngram-fsts --enable-lookahead-fsts --with-pic"

BUILD_FOLDER=build

INSTALL_PREFIX=$1

if [ ! $# == 1 ]; then
    echo "[ERROR] Usage: $0 <install-prefix>"
    exit -1
fi

if [ ! -f ${BUILD_FOLDER}/openfst-${OPENFST_VERSION}.tar.gz ]; then
    ${WGET} -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz || \
            ${WGET} -T 10 -t 3 -c https://www.openslr.org/resources/2/openfst-${OPENFST_VERSION}.tar.gz;
    mv openfst-${OPENFST_VERSION}.tar.gz ${BUILD_FOLDER}/openfst-${OPENFST_VERSION}.tar.gz
fi

(
    cd build/
    
    rm -R openfst-${OPENFST_VERSION}
    tar -xvf openfst-${OPENFST_VERSION}.tar.gz
    (
        openfst_add_CXXFLAGS="-g -O2"

        cd openfst-${OPENFST_VERSION}

        if [ ! -d ${INSTALL_PREFIX} ]; then
            mkdir -p ${INSTALL_PREFIX}
        fi

        ./configure --prefix=`pwd` ${OPENFST_CONFIGURE} CXX="${CXX}" --prefix=${INSTALL_PREFIX}\
            CXXFLAGS="${CXXFLAGS} ${openfst_add_CXXFLAGS}" LDFLAGS="${LDFLAGS}" LIBS="-ldl"
        make -j ${NUM_JOB}
        make install
    )
)


