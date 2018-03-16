#!/bin/bash

# Installs beamformit from the location https://github.com/xanguera/BeamformIt

# libsndfile needed by beamformit
[ ! -f libsndfile-1.0.25.tar.gz ] && \
  wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
[ ! -d libsndfile-1.0.25 ] && \
  tar xzf libsndfile-1.0.25.tar.gz
(
  cd libsndfile-1.0.25
  ./configure --prefix=$PWD
  make
  make install
)

# building beamformit
[ ! -d ./BeamformIt ] &&
  git clone https://github.com/xanguera/BeamformIt
(
  cd BeamformIt
  git pull
  cmake -DLIBSND_INSTALL_DIR=$PWD/../libsndfile-1.0.25 .
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
