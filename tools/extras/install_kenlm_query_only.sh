#!/bin/bash
#
# 2020 author Jiayu DU
#
# this script downloads latest kenlm and compile it in "query only" mode
# "query only" mode means you can only use this kenlm lib for runtime lm score query.
#
# NOTES:
# we don't want a full-build of kenlm inside kaldi repo because:
# kenlm's full-build (with arpa counting/smoothing/interpolation supports) requires EIGEN and BOOST, 
# it's impractical to incorperate such heavy libraries into Kaldi repo.
# the purpose of this script is to provide just enough support,
# to use kenlm as a runtime lm scorer in Kaldi, without arpa training stuff.
#
# If you DO want to use kenlm to "train" an arpa from scratch,
# you need to setup these dependencies yourself via system package manager / your admin,
# or compile from source code, following kenlm's self-contained CMAKE building system.
# after that, we provide a demo kenlm training script at: egs/wsj/s5/utils/train_arpa_with_kenlm.sh
# if you come across any problems in installing full-build kenlm, 
# the first place to ask is (https://github.com/kpu/kenlm), not kaldi.
# 
# LEGAL STUFF:
# KENLM codes are intactly downladed into tools/kenlm dir, together with its LGPL LICENSE, 
# it's not as free as Kaldi's APACHE2.0, KALDI users should be aware of this,
# and it is the users responsibility to follow these open-source software licenses.

echo "****() Installing KenLM (QUERY ONLY mode)"

if [ ! -d kenlm ]; then
    echo "No exisiting kenlm, try to clone the latest repo from https://github.com/kpu/kenlm"
    if which git >&/dev/null; then
        git clone https://github.com/kpu/kenlm.git
    else
        echo "no git installed? need it for cloning source code from github."
        exit 1;
    fi
fi


# now compile query-only kenlm library, to be linked with Kaldi
# the following code are based on compile_query_only.sh in kenlm source code
cd kenlm
rm {lm,util}/*.o 2>/dev/null
set -e

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

echo "KenLM (QUERY ONLY) successfully compile into $KALDI_ROOT/tools/kenlm/libkenlm.so"
echo "To link it, add following flags to your compiler-linker toolchain:"
echo "  -I$KALDI_ROOT/tools/kenlm/  -DKENLM_MAX_ORDER=6  -Wl,-rpath,$KALDI_ROOT/tools/kenlm/  -L$KALDI_ROOT/tools/kenlm/  -lkenlm"
