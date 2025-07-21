#!/usr/bin/env bash

VER=1.10

WGET=${WGET:-wget}

if [ ! -f liblbfgs-$VER.tar.gz ]; then
  if [ -d "$DOWNLOAD_DIR" ]; then
    cp -p "$DOWNLOAD_DIR/liblbfgs-$VER.tar.gz" . || exit 1
  else
    $WGET https://github.com/downloads/chokkan/liblbfgs/liblbfgs-$VER.tar.gz || exit 1
  fi
fi

tar -xzf liblbfgs-$VER.tar.gz
cd liblbfgs-$VER
./configure --prefix=`pwd`
make
# due to the liblbfgs project directory structure, we have to use -i
# but the erros are completely harmless
make -i install
cd ..

(
  [ ! -z "${LIBLBFGS}" ] && \
    echo >&2 "LIBLBFGS variable is aleady defined. Undefining..." && \
    unset LIBLBFGS

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${LIBLBFGS}" ] && \
    echo >&2 "libLBFGS config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export LIBLBFGS=$wd/liblbfgs-1.10"
  echo export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}':'${LIBLBFGS}'/lib/.libs
) >> env.sh

