#!/bin/bash

# We take into account dependency pointers optionally set in the enironment.
# Typical usage shown below; any one can be safely left unset.
#   INCDIRS="~/xroot/usr/include"
#   LIBDIRS="~/xroot/usr/lib /usr/lib/openblas-base"
#   CXX=gcc++-4.9
#   CFLAGS="-march=native -O2"
#   LDFLAGS="-llapack"

# Maximum make parallelism. Simply -j runs out of memory on Travis VM.
MAXPAR=3

# Run verbose (run and echo) and exit if failed.
runvx() {
  echo "\$ $@"
  "$@" || exit 1
}

# $(addsw -L foo bar) => "-Lfoo -Lbar".
addsw() {
  local v=() s=$1; shift;
  for d; do v+=("$s$d"); done
  echo ${v[@]};
}

# $(mtoken CXX gcc) => "CXX=gcc"; # $(mtoken CXX ) => "".
mtoken() { echo ${2+$1=$2}; }

# Print machine info and environment.
runvx uname -a
runvx env

# Prepare make command fragments.
CF="$CFLAGS -g $(addsw -I $INCDIRS)"
LDF="$LDFLAGS $(addsw -L $LIBDIRS)"
CCC="$(mtoken CC $CXX) $(mtoken CXX $CXX)"

runvx cd tools
runvx make openfst $CCC CXXFLAGS="$CF" -j$MAXPAR
cd ..
runvx cd src
runvx ./configure --use-cuda=no  --mathlib=OPENBLAS --openblas-root=$XROOT/usr

make_kaldi() {
  runvx make "$@" $CCC EXTRA_CXXFLAGS="$CF" EXTRA_LDLIBS="$LDF"
}

#make_kaldi mklibdir base matrix -j$MAXPAR
#make_kaldi matrix/test

make_kaldi all -j$MAXPAR
make_kaldi test
