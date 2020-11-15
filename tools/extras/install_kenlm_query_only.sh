#!/bin/bash
#
# 2020 author Jiayu DU
#
# this script downloads latest code from official kenlm repo (https://github.com/kpu/kenlm)
# and compile it partially, as a runtime library for lm scoring in Kaldi (*query only* mode),
# to be specific, "query only" mode generates:
# 
# 1. executables: $KALDI_ROOT/tools/kenlm/bin/{build_binary,query}
#    * "build_binary" is used to compile ngram apra files into kenlm model files
#      (demo script: $KALDI_ROOT/egs/wsj/s5/utils/build_kenlm_model_from_arpa.sh)
#    * "query" is used to score plain text(one line per sentence) with kenlm models
#
# 2. library: $KALDI_ROOT/tools/kenlm/libkenlm.so
#    along with this script, run $KALDI_ROOT/src/configure with "--enable-kenlm" option,
#    configure will generate KenLM related compiler flags into "kaldi.mk", which are:
#    * KENLM_ROOT =  $KALDI_ROOT/tools/kenlm/
#    * KENLM_CXXFLAGS = -DHAVE_KENLM  -I$KALDI_ROOT/tools/kenlm/  -DKENLM_MAX_ORDER=6
#    * KENLM_LDFLAGS = -Wl,-rpath,$KALDI_ROOT/tools/kenlm/  -L$KALDI_ROOT/tools/kenlm/  -lkenlm
#    use these variables in any Kaldi submodule's Makefile as you like.
#    In Kaldi, instead of interacting with KenLM's public interfaces,
#    you should use KenLM's FST wrapper class in $KALDI_ROOT/src/lm/kenlm.{h,cc}
#
# Note that we currently don't have a full-build of KenLM inside Kaldi,
# because a full-build of KenLM (with arpa counting/smoothing/interpolation supports) depends 
# on EIGEN and BOOST, too heavy to integrate.
#
# If you DO want to use KenLM to "train" an arpa from scratch,
# you need to setup these dependencies yourself via system package manager / contact your admin,
# and compile with the self-contained CMAKE system in KenLM.
# After that, you can refer to the kenlm training demo script at: egs/wsj/s5/utils/train_arpa_with_kenlm.sh
# if you come across any problems installing full-build kenlm, 
# the first place to ask is (https://github.com/kpu/kenlm), not kaldi.
# 
# LEGAL STUFF:
# KenLM codes are intactly cloned into tools/kenlm dir, together with its LGPL LICENSE, 
# it's not as free as Kaldi's Apache-2.0, Kaldi users should be aware of this,
# and it is the users responsibility to comply with these open-source licenses.

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

# now compile query-only kenlm library (to be linked with Kaldi)
# the following code are based on kenlm/compile_query_only.sh
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

# register kenlm binary path to env.sh
KENLM_ROOT=$(pwd)/kenlm
echo "export KENLM_ROOT=$KENLM_ROOT" >> env.sh
echo "export PATH=\${PATH}:\${KENLM_ROOT}/bin" >> env.sh

echo "KenLM (QUERY ONLY mode) successfully installed in tools/kenlm/"
echo "to use kenlm runtime in kaldi, pass --enable-kenlm  option to configure,"
echo "and read usage notes at the beginning of $0"
