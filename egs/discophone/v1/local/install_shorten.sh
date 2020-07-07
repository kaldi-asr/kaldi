#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Piotr Żelasko)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# http://shnutils.freeshell.org/shorten/
# shorten is a fast, low complexity waveform coder (i.e. audio compressor),
# originally written by Tony Robinson at SoftSound.
# It can operate in both lossy and lossless modes.

if command -v shorten; then
  echo "shorten already available in PATH: skipping setup."
  exit 0
fi

wget http://shnutils.freeshell.org/shorten/dist/src/shorten-3.6.1.tar.gz
tar xf shorten-3.6.1.tar.gz
cd shorten-3.6.1
echo "Compiling shorten..."
./configure &>/dev/null
make &>/dev/null
echo "Shorten is built!"
