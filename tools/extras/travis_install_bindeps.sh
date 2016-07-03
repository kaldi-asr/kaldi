#!/bin/bash

set -e

xroot=${1:-~/xroot}

mkdir -p $xroot
cd $xroot

add_deb () {
  echo "Adding deb package $1 to $xroot"
  wget -nv $1
  dpkg-deb -x ${1##*/} $xroot
}

# OpenBLAS and Netlib LAPACK binaries from Trusty.
add_deb http://mirrors.kernel.org/ubuntu/pool/main/l/lapack/liblapacke-dev_3.5.0-2ubuntu1_amd64.deb
add_deb http://mirrors.kernel.org/ubuntu/pool/main/l/lapack/liblapacke_3.5.0-2ubuntu1_amd64.deb
add_deb http://mirrors.kernel.org/ubuntu/pool/universe/o/openblas/libopenblas-dev_0.2.8-6ubuntu1_amd64.deb
add_deb http://mirrors.kernel.org/ubuntu/pool/universe/o/openblas/libopenblas-base_0.2.8-6ubuntu1_amd64.deb
add_deb http://mirrors.kernel.org/ubuntu/pool/main/b/blas/libblas-dev_1.2.20110419-7_amd64.deb
add_deb http://mirrors.kernel.org/ubuntu/pool/main/b/blas/libblas3_1.2.20110419-7_amd64.deb

# Show extracted packages.
find $xroot | sort
