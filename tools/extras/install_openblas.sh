#!/usr/bin/env bash

OPENBLAS_VERSION=0.3.13

WGET=${WGET:-wget}

set -e

if ! command -v gfortran 2>/dev/null; then
  echo "$0: gfortran is not installed.  Please install it, e.g. by:"
  echo " apt-get install gfortran"
  echo "(if on Debian or Ubuntu), or:"
  echo " yum install gcc-gfortran"
  echo "(if on RedHat/CentOS).  On a Mac, if brew is installed, it's:"
  echo " brew install gfortran"
  exit 1
fi

tarball=OpenBLAS-$OPENBLAS_VERSION.tar.gz

rm -rf OpenMathLib-OpenBLAS-* OpenBLAS OpenBLAS-* OpenBLAS-*.tar.gz

if [ -n "${DOWNLOAD_DIR:-}" ] && [ -d "$DOWNLOAD_DIR" ]; then
  cp -p "$DOWNLOAD_DIR/$tarball" .
else
  url="https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v${OPENBLAS_VERSION}.tar.gz"
  $WGET -t3 -nv -O "$tarball" "$url"
fi

tar xzf "$tarball"

src_dir=$(tar tzf "$tarball" | head -1 | cut -d/ -f1)
mv "$src_dir" OpenBLAS

make PREFIX=$(pwd)/OpenBLAS/install USE_LOCKING=1 USE_THREAD=0 -C OpenBLAS all install

echo "OpenBLAS is installed successfully."
rm "$tarball"
