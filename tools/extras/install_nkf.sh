#!/bin/bash

# Copyright 2017  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

set -u
set -e

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You mush call this script from the tools/ directory" && exit 1;

echo Downloading and installing the nkf tools
#Download
if [ ! -f nkf-2.1.4.tar.gz ]; then
  wget https://osdn.net/dl/nkf/nkf-2.1.4.tar.gz
  tar -vxzf nkf-2.1.4.tar.gz
fi

#install
cd nkf-2.1.4
make
cd ..

#add to env.sh
if [ -f env.sh ]; then
  wd=`pwd`
  echo "export NKF=$wd/nkf-2.1.4" >> env.sh
  echo "export PATH=\${PATH}:\${NKF}" >> env.sh
fi
echo Done making the nkf tools

echo >&2 "Installation of nkf finished successfully"
echo >&2 "Please source the tools/env.sh in your path.sh to enable it"

