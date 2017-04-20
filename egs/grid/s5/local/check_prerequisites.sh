#!/bin/bash

# Copyright 2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
#           
# Apache 2.0.
#
# This script checks the prerequisite for running the experiments and creates the required files and directories.

# Check wave file directory
if [ ! -d $WAV_ROOT ]; then
  echo "Cannot find audio directory $WAV_ROOT."
  echo "Please download and extract the CHiME-1 or CHiME-2 Challenge Data first or adjust the provided audio directory."
  echo ""
  echo "CHiME-1 download:"
  echo "  train set  http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_train_reverberated.tar.gz"
  echo "  devel set  http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_devel_isolated.tar.gz"
  echo "  test set   http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_test_isolated.tar.gz"
  echo ""
  echo "CHiME-2 download:"
  echo "  train set ftp://ftp.dcs.shef.ac.uk/share/spandh/chime_challenge/grid/train_isolated.tgz"
  echo "  devel set ftp://ftp.dcs.shef.ac.uk/share/spandh/chime_challenge/grid/devel_isolated.tgz"
  echo "  test set 	ftp://ftp.dcs.shef.ac.uk/share/spandh/chime_challenge/grid/test_isolated.tgz"
  exit 1;
fi

if [ -z "$VIDEO_ROOT" ]; then
  export VIDEO_ROOT="$REC_ROOT/video"
  if [ ! -d $VIDEO_ROOT ]; then
    mkdir -p $VIDEO_ROOT
    wget -P $VIDEO_ROOT https://zenodo.org/record/260211/files/chime2_track1_isolated_video_25_10.tar.gz || exit 1;

    echo "Extracting video files."
    tar -xzf $VIDEO_ROOT/chime2_track1_isolated_video_25_10.tar.gz -C $VIDEO_ROOT || exit 1;
  fi
else
  if [ ! -d $VIDEO_ROOT ]; then
    echo "Cannot find video directory $VIDEO_ROOT."
    echo "Please unset video directory in path.sh or specify a valid location."
    exit 1;
  fi
fi

# Setup symlinks for steps and utils
if [ ! -e "steps" ]; then
  echo "./steps does not exist. Creating symlink."
  ln -s $KALDI_ROOT/egs/wsj/s5/steps steps || exit 1
fi

if [ ! -e "utils" ]; then
  echo "./utils does not exist. Creating symlink..."
  ln -s $KALDI_ROOT/egs/wsj/s5/utils utils || exit 1
fi
