#!/usr/bin/env bash

OPENBLAS_VERSION=0.3.20
CLAPACK_VERSION=3.2.1

git clone -b v${OPENBLAS_VERSION} --single-branch https://github.com/xianyi/OpenBLAS
git clone -b v${CLAPACK_VERSION} --single-branch https://github.com/alphacep/clapack \

make -C OpenBLAS ONLY_CBLAS=1 DYNAMIC_ARCH=1 TARGET=NEHALEM USE_LOCKING=1 USE_THREAD=0 NUM_THREADS=512 all
make -C OpenBLAS PREFIX=$(pwd)/OpenBLAS/install install
mkdir -p clapack/BUILD && cd clapack/BUILD && cmake .. \
    && make -j 10 -C F2CLIBS \
    && make -j 10 -C BLAS \
    && make -j 10 -C SRC \
    && find . -name "*.a" | xargs cp -t ../../OpenBLAS/install/lib
