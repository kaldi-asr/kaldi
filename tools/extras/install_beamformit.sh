#!/usr/bin/env bash

LIBSNDFILE_VERSION=1.0.25

GIT=${GIT:-git}
WGET=${WGET:-wget}

# Installs beamformit from the location https://github.com/xanguera/BeamformIt

# libsndfile needed by beamformit
if [ ! -f libsndfile-$LIBSNDFILE_VERSION.tar.gz ]; then
  if [ -d "$DOWNLOAD_DIR" ]; then
    cp -p "$DOWNLOAD_DIR/libsndfile-$LIBSNDFILE_VERSION.tar.gz" . || exit 1
  else
    $WGET http://www.mega-nerd.com/libsndfile/files/libsndfile-$LIBSNDFILE_VERSION.tar.gz || exit 1
  fi
fi
[ ! -d libsndfile-$LIBSNDFILE_VERSION ] && \
  tar xzf libsndfile-$LIBSNDFILE_VERSION.tar.gz
(
  cd libsndfile-$LIBSNDFILE_VERSION
  ./configure --prefix=$PWD
  make
  make install
)

# building beamformit
[ ! -d ./BeamformIt ] &&
  $GIT clone https://github.com/xanguera/BeamformIt
(
  cd BeamformIt
  $GIT pull
  cmake -DLIBSND_INSTALL_DIR=$PWD/../libsndfile-$LIBSNDFILE_VERSION .
  make
)

# add config into env.sh
(
  [ ! -z "${BEAMFORMIT}" ] && \
    echo >&2 "BEAMFORMIT variable is aleady defined. Undefining..." && \
    unset BEAMFORMIT

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${BEAMFORMIT}" ] && \
    echo >&2 "BeamformIt config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export BEAMFORMIT=$wd/BeamformIt"
  echo "export PATH=\${PATH}:\${BEAMFORMIT}"
) >> env.sh
