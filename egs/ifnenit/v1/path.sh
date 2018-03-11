#!/bin/bash

# path to Kaldi's root directory
export KALDI_ROOT=`pwd`/../../..

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export LD_LIBRARY_PATH=/home/dpovey/libs:$KALDI_ROOT/src/chainbin:/usr/local/lib:$LD_LIBRARY_PATH
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
