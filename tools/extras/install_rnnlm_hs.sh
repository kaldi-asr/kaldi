#!/bin/bash

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

echo "Installing RNNLM-HS 0.1b"

cd rnnlm-hs-0.1b
make
