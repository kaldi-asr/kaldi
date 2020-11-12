#!/bin/bash
# this script downloads latest kenlm and compile it in "query only" mode
# "query only" mode means you can only use this kenlm lib for runtime lm score query.
# 

echo "****() Installing KenLM (QUERY ONLY mode)"

if [ ! -d kenlm ]; then
    echo "No exisiting kenlm.tar.gz, try to clone the latest repo from https://github.com/kpu/kenlm"
    if which git >&/dev/null; then
        #wget -T 10 -t 3 -c --no-check-certificate https://kheafield.com/code/kenlm/kenlm.tar.gz
        git clone https://github.com/kpu/kenlm.git
    else
        echo "no git installed? need it for cloning source code from github."
        exit 1;
    fi
fi

# now compile query-only kenlm library, to be linked with Kaldi
# the following code are based on compile_query_only.sh in kenlm/
cd kenlm
rm {lm,util}/*.o 2>/dev/null
#set -e

CXX=${CXX:-g++}
# for .a
#CXXFLAGS+=" -I. -O3 -DNDEBUG -DKENLM_MAX_ORDER=6"
# for .so
CXXFLAGS+=" -I. -O3 -DNDEBUG -DKENLM_MAX_ORDER=6 -fPIC"

echo 'Compiling with '$CXX $CXXFLAGS
#Grab all cc files in these directories except those ending in test.cc or main.cc
objects=""
for i in util/double-conversion/*.cc util/*.cc lm/*.cc $ADDED_PATHS; do
  if [ "${i%test.cc}" == "$i" ] && [ "${i%main.cc}" == "$i" ]; then
    $CXX $CXXFLAGS -c $i -o ${i%.cc}.o
    objects="$objects ${i%.cc}.o"
  fi
done

mkdir -p bin
if [ "$(uname)" != Darwin ]; then
  CXXFLAGS="$CXXFLAGS -lrt"
fi
$CXX lm/build_binary_main.cc $objects -o bin/build_binary $CXXFLAGS $LDFLAGS
$CXX lm/query_main.cc $objects -o bin/query $CXXFLAGS $LDFLAGS

# for .a
#ar -crv libkenlm.a $objects

# for .so
$CXX $objects -shared -o libkenlm.so
cd ..

KENLM_ROOT=`pwd`/kenlm
echo "export KENLM_ROOT=$KENLM_ROOT" >> env.sh

echo "KenLM (QUERY ONLY) successfully compile into dynamic library $KALDI_ROOT/tools/kenlm/libkenlm.so"
echo "To link against the lib, you need to add following flags to your compiler-linker toolchain:"
echo "  -I$KALDI_ROOT/tools/kenlm/  -DKENLM_MAX_ORDER=6  -Wl,-rpath,$KALDI_ROOT/tools/kenlm/  -L$KALDI_ROOT/tools/kenlm/  -lkenlm"
