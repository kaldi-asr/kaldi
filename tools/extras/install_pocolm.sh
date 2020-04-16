#!/usr/bin/env bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

GIT=${GIT:-git}

set -u
set -e


# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;

if [ -d pocolm ]; then
  echo "$0: Assuming pocolm is already installed Please delete the directory"
  echo "./pocolm if you need to reinstall."
  exit 0
fi

echo Downloading and installing the pocolm tools
$GIT clone https://github.com/danpovey/pocolm.git || exit 1;
cd pocolm/src
make || exit 1;
echo Done making the pocolm tools
cd ../..

