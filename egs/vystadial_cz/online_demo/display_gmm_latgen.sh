#!/usr/bin/env bash

# source the settings
. ./path.sh

for n in `cut -d' ' -f1 $wav_scp` ; do
    utils/show_lattice.sh --mode save --format svg $n $lattice $wst
done
