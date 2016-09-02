#!/bin/bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

set -u
set -e


# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You must call this script from the tools/ directory" && exit 1;

echo Downloading and installing the pocolm tools
git clone https://github.com/danpovey/pocolm.git || exit 1;
cd pocolm/src
make || exit 1;
echo Done making the pocolm tools
cd ../..

