#!/bin/bash

# http://www.speech.sri.com/projects/srilm/download.html

if [ ! -f srilm.tgz ]; then
  echo This script cannot install SRILM in a completely automatic
  echo way because you need to put your address in a download form.
  echo Please download SRILM from http://www.speech.sri.com/projects/srilm/download.html
  echo put it in ./srilm.tgz, then run this script.
fi

! which gawk 2>/dev/null && \
   echo "GNU awk is not installed so SRILM will probably not work correctly: refusing to install" && exit 1;

mkdir -p srilm
cd srilm
tar -xvzf ../srilm.tgz

# set the SRILM variable in the top-level Makefile to this directory.
cp Makefile tmpf

cat tmpf | awk -v pwd=`pwd` '/SRILM =/{printf("SRILM = %s\n", pwd); next;} {print;}' \
  > Makefile || exit 1;

make

