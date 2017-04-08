#!/bin/bash

# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  John Hopkins University (author: Daniel Povey)
# Apache 2.0

mkdir -p db

cd db  ### Note: the rest of this script is executed from the directory 'db'.

# TED-LIUM database:
if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
  if [ ! -e TEDLIUM_release2 ]; then
    ln -sf /export/corpora5/TEDLIUM_release2
  fi
  echo "$0: linking the TEDLIUM data from /export/corpora5/TEDLIUM_release2"
else
  if [ ! -e TEDLIUM_release2 ]; then
    echo "$0: downloading TEDLIUM_release2 data (it won't re-download if it was already downloaded.)"
    # the following command won't re-get it if it's already there
    # because of the --continue switch.
    wget --continue http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz || exit 1
    tar xf "TEDLIUM_release2.tar.gz"
  else
    echo "$0: not downloading or un-tarring TEDLIUM_release2 because it already exists."
  fi
fi


num_sph=$(find TEDLIUM_release2/ -name '*.sph' | wc -l)
if [ "$num_sph" != 1514 ]; then
  echo "$0: expected to find 1514 .sph files in the directory db/TEDLIUM_release2, found $num_sph"
  exit 1
fi

exit 0

