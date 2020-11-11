#!/usr/bin/env bash

# A patch for sctk-2.4.0 smooth installation under Cygwin

os=`uname -a | awk '{printf $NF}'`

if [ "$os" == "Cygwin" ]
then
  cp src/rfilter1/makefile.in src/rfilter1/makefile.in.orig
  sed 's/OPTIONS=-DNEED_STRCMP=1/OPTIONS=/g' src/rfilter1/makefile.in > tmpf
  mv tmpf src/rfilter1/makefile.in   
fi
