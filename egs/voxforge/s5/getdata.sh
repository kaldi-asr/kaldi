#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

# Downloads and extracts the data from VoxForge website

# defines "DATA_ROOT" variable - the location to store data 
source path.sh

DATA_SRC="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"
DATA_TGZ=${DATA_ROOT}/tgz
DATA_EXTRACT=${DATA_ROOT}/extracted

# delete the archive after it's extracted?
deltgz=false

# give the user opportunity to change behaviour using cmd line - e.g. --deltgz true
source utils/parse_options.sh

mkdir -p ${DATA_TGZ} 2>/dev/null

# Check if the executables needed for this script are present in the system
command -v wget >/dev/null 2>&1 ||\
 { echo "\"wget\" is needed but not found"'!'; exit 1; }

echo "--- Starting VoxForge data download (may take some time) ..."
wget -P ${DATA_TGZ} -l 1 -N -nd -c -e robots=off -A tgz -r -np ${DATA_SRC} || \
 { echo "WGET error"'!' ; exit 1 ; }

echo "--- Starting VoxForge archives extraction ..."
for a in ${DATA_TGZ}/*.tgz; do
  tar -C ${DATA_EXTRACT} -xf $a
done

if ${deltgz}; then
  rm -rf ${DATA_TGZ}
fi
