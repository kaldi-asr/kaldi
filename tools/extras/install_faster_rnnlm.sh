#!/usr/bin/env bash

# The script downloads and installs faster-rnnlm
# https://github.com/yandex/faster-rnnlm

GIT=${GIT:-git}

set -e

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

echo "Installing Faster RNNLM"

if [ ! -d "faster-rnnlm" ]; then
    $GIT clone https://github.com/yandex/faster-rnnlm.git
fi

cd faster-rnnlm
$GIT pull
./build.sh
ln -sf faster-rnnlm/rnnlm
