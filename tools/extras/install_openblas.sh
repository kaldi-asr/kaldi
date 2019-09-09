#!/bin/bash


OPENBLAS_VERSION=0.3.5

set -e

if ! command -v gfortran 2>/dev/null; then
  echo "$0: gfortran is not installed.  Please install it, e.g. by:"
  echo " apt-get install gfortran"
  echo "(if on Debian or Ubuntu), or:"
  echo " yum install fortran"
  echo "(if on RedHat/CentOS).  On a Mac, if brew is installed, it's:"
  echo " brew install gfortran"
  exit 1
fi


rm -rf xianyi-OpenBLAS-* OpenBLAS

wget -t3 -nv -O- $(wget -qO- "https://api.github.com/repos/xianyi/OpenBLAS/releases/tags/v${OPENBLAS_VERSION}" | python -c 'import sys,json;print(json.load(sys.stdin)["tarball_url"])') | tar xzf -

mv xianyi-OpenBLAS-* OpenBLAS

make PREFIX=$(pwd)/OpenBLAS/install USE_THREAD=0 -C OpenBLAS all install
