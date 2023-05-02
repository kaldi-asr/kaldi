#!/usr/bin/env bash

WGET=${WGET:-wget}

# The script automatically choose default settings of miniconda for installation
# Miniconda will be installed in the HOME directory. ($HOME/miniconda3).
# Also don't make miniconda's python as default.

if [ -d "$DOWNLOAD_DIR" ]; then
  cp -p "$DOWNLOAD_DIR/Miniconda3-latest-Linux-x86_64.sh" . || exit 1
else
  $WGET https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh || exit 1
fi
bash Miniconda3-latest-Linux-x86_64.sh -b

$HOME/miniconda3/bin/python -m pip install --user tqdm
$HOME/miniconda3/bin/python -m pip install --user scikit-learn
$HOME/miniconda3/bin/python -m pip install --user librosa
$HOME/miniconda3/bin/python -m pip install --user h5py
