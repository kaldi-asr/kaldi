#!/bin/bash

# path to Kaldi's root directory
root=`pwd`/../../..

export PATH=${root}/src/bin:${root}/tools/openfst/bin:${root}/src/fstbin/:${root}/src/gmmbin/:${root}/src/featbin/:${root}/src/fgmmbin:${root}/src/sgmmbin:${root}/src/lm:${root}/src/latbin:${root}/src/tiedbin/:$PATH  

# path to the directory in which the subset of RM corpus is stored
export RM1_ROOT=`pwd`/data/download

export LC_ALL=C
export LC_LOCALE_ALL=C

