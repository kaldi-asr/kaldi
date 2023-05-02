#!/bin/bash
#
# Copyright 2021 Hang Lyu
#           2021 kkm
#
# This script attempts to install tcmalloc.
# The tcmalloc provides the more efficient way to malloc so that it can speed
# up the code, especially for the decoding code which contains massive memory
# allocation operations.
#
# At the same time, the tcmalloc also provides some profilers which can help
# the user to analysis the performance of the code. 
# However, there may have some deadlock problems when you use the profilers with
# default build-in glibc. So the libunwind is recommanded to be installed. When
# the deadlock problems happen, the user can try to link with the library
# libunwind. But there may still be some crash on x64 platform. (Link with
# "-lunwind" after your installation if you want to include it.)
# As very rare end users will need it and we believe the users who need it must
# be qualified to reconfigure and build as much of toolset as they want, we skip
# the installation around libunwind.
#
# Depending on different platforms which are used by different users, the users
# also can try differnet malloc libraries such as tbbmalloc, ptmalloc and so on.
# From our test, the tcmalloc is most efficient on our platform.

set -e

# Make sure we are in the tools/ directory.
if [ $(basename $PWD) == extras ]; then
  cd ..
fi

! [ $(basename $PWD) == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;

# prepare tcmalloc
if [ -d gperftools ]; then
  echo "$0: existing 'gperftools' subdirectory is renamed 'gperftools.bak'"
  mv -f gperftools gperftools.bak
fi

wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz &&
  tar xzf gperftools-2.9.1.tar.gz &&
  mv gperftools-2.9.1 gperftools

# install tcmalloc
(
  cd gperftools &&
  ./configure --prefix=$PWD --enable-minimal --disable-debugalloc --disable-static &&
  make &&
  make install
)
