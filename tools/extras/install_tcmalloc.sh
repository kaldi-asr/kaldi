#!/bin/bash
#
# Copyright 2021 Hang Lyu
#
# This script attempts to install tcmalloc and libunwind.
# The tcmalloc provides the more efficient way to malloc so that it can speed
# up the code, especially for the decoding code which contains massive memory
# allocation operations.
#
# At the same time, the tcmalloc also provides some profilers which can help
# the user to analysis the performance of the code. However, there may have 
# some deadlock problems when you use the profilers with default build-in glibc.
# So the libunwind is recommanded to be installed. When the deadlock problems
# happen, the user can try to link with the library libunwind. But there may
# still be some crash on x64 platform. You can skip the installation about
# libunwind if you don't need it. (Link with "-lunwind" after your installation
# if you want to include it.)
#
# Depending on different platforms which are used by different users, the users
# also can try differnet malloc libraries such as tbbmalloc and so on. From our
# test, the tcmalloc is most efficient on our platform.


# Make sure we are in the tools/ directory.
if [ $(basename $PWD) == extras ]; then
  cd ..
fi

! [ $(basename $PWD) == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;

# prepare libunwind
echo "****() Installing libunwind"
if [ ! -e libunwind-1.5-rc1.tar.gz ]; then
  echo "Trying to download libunwind via wget"

  if ! which wget >&/dev/null; then
    echo "This script requires you to first install wget"
    echo "You can also just download libunwind-1.5-rc1.tar.gz from"
    echo "http://download-mirror.savannah.gnu.org/releases/libunwind/libunwind-1.5-rc1.tar.gz"
    exit 1;
  fi

  wget http://download.savannah.nongnu.org/releases/libunwind/libunwind-1.5-rc1.tar.gz || exit 1;

  if [ ! -f libunwind-1.5-rc1.tar.gz ]; then
    echo "Download of libunwind failed!"
    echo "Aborting script. Please download and install libunwind manually!"
    exit 1;
  fi
fi

# install libunwind
# The installation directory can be changed. But be careful, set the prefix as 
#`pwd` will cause recursive problem.
tar vxzf libunwind-1.5-rc1.tar.gz
cd libunwind-1.5-rc1
./configure --prefix=$PWD/install || exit 1;
make || exit 1;
make install || exit 1;
cd ..

# prepare tcmalloc
if [ -d gperftools ]; then
  echo "$0: Assuming gperftools is already installed Please delete the directory"
  echo "./gperftools if you need to reinstall."
  mv gproftools gproftools_backup
fi
git clone https://github.com/gperftools/gperftools.git gperftools

# install tcmalloc
wd=$PWD
cd gperftools
bash autogen.sh
./configure --prefix=$PWD || exit 1;
make || exit 1;
make install || exit 1;
cd ..

# add path
(
  wd=$PWD
  echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$wd/gperftools/lib
  echo export PATH=$PATH:$wd/gperftools/bin
) >> env.sh


