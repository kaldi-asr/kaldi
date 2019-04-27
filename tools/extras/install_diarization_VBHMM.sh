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
  git clone https://github.com/GoVivaceInc/VB_diarization  
  cp VB_diarization/VB_diarization.py ../egs/callhome_diarization/v1/local/
fi

pip install numexpr
