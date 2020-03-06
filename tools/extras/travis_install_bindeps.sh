#!/usr/bin/env bash

WGET=${WGET:-wget}

set -e

xroot=${1:-~/xroot}

mkdir -p $xroot
cd $xroot

add_deb () {
  echo "Adding deb package $1 to $xroot"
  $WGET -nv $1
  dpkg-deb -x ${1##*/} $xroot
}

# OpenBLAS and Netlib LAPACK binaries from Trusty.
add_deb https://mirrors.kernel.org/ubuntu/pool/main/l/lapack/liblapacke-dev_3.5.0-2ubuntu1_amd64.deb
add_deb https://mirrors.kernel.org/ubuntu/pool/main/l/lapack/liblapacke_3.5.0-2ubuntu1_amd64.deb
add_deb https://mirrors.kernel.org/ubuntu/pool/universe/o/openblas/libopenblas-dev_0.2.8-6ubuntu1_amd64.deb
add_deb https://mirrors.kernel.org/ubuntu/pool/universe/o/openblas/libopenblas-base_0.2.8-6ubuntu1_amd64.deb
add_deb https://mirrors.kernel.org/ubuntu/pool/main/b/blas/libblas-dev_1.2.20110419-7_amd64.deb
add_deb https://mirrors.kernel.org/ubuntu/pool/main/b/blas/libblas3_1.2.20110419-7_amd64.deb

if [[ "$(ccache --version 2>/dev/null | sed -n '1{s/^[a-z ]*//;s/\./0/g;p}')" -lt 30304 ]]; then
    add_deb https://mirrors.kernel.org/debian/pool/main/c/ccache/ccache_3.3.4-1_amd64.deb
fi

# Show extracted package files.
find $xroot | sort
