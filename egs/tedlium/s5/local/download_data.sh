#!/bin/bash

# Copyright  2014 Nickolay V. Shmyrev 
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

pushd db
# TED-LIUM database:
if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
 ln -s /export/corpora5/TEDLIUM_release1
else
  if [ ! -f TEDLIUM_release1.tar.gz ]; then
    wget http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz || exit 1
    tar xf TEDLIUM_release1.tar.gz
  fi
fi
# Generic CMU language model:
if [ ! -f cmusphinx-5.0-en-us.lm.gz ]; then
    wget \
        http://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English%20Generic%20Language%20Model/cmusphinx-5.0-en-us.lm.gz/download \
        -O cmusphinx-5.0-en-us.lm.gz || exit 1
fi

popd
