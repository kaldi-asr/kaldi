#!/bin/bash

# Copyright 2017  Johns Hopkins University (author: Daniel Povey)
#           2017  Hang Lyu
# Apache 2.0

# nkf(Network Kanji Filter) is a kanji code converter. It converts input kanji
# code to designated kanji code such as ISO-2022-JP, UTF-8, and so on.
# In kaldi, it will be used in egs/csj. (Corpus of Spontaneous Japanese data)

set -u
set -e

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
  echo "You mush call this script from the tools/ directory" && exit 1;

echo Downloading and installing the nkf tools
#Download
if [ ! -f nkf-2.1.4.tar.gz ]; then
  wget https://osdn.net/dl/nkf/nkf-2.1.4.tar.gz
  tar -vxzf nkf-2.1.4.tar.gz
fi

#install
cd nkf-2.1.4
make
cd ..

#add to env.sh
if [ -f env.sh ]; then
  wd=`pwd`
  echo "export NKF=$wd/nkf-2.1.4" >> env.sh
  echo "export PATH=\${PATH}:\${NKF}" >> env.sh
fi
echo Done making the nkf tools

echo >&2 "Installation of nkf finished successfully"
echo >&2 "Please source the tools/env.sh in your path.sh to enable it"

