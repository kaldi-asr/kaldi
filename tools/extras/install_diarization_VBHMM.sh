#!/bin/bash
set -u
set -e


# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
    cd ..
fi

! [ `basename $PWD` == tools ] && \
    echo "You must call this script from the tools/ directory" && exit 1;

# We download the original VB HMM scripts of the Brno University of Technology.
# numexpr is a required dependency for speeding up the VB_diarization.
if [ ! -d VB_diarization ]; then
  wget  http://www.fit.vutbr.cz/~burget/VB_diarization.zip
  unzip VB_diarization.zip
fi

pip install numexpr
