#!/bin/bash

if [ -z ${KALDI_ROOT+x} ]; then
  export KALDI_ROOT=`pwd -P`/../../..
fi
if [ -z ${LD_LIBRARY_PATH+x} ]; then
  export LD_LIBRARY_PATH=
fi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/bin/nvcc
export LC_ALL=C
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/hhadian/dev/behavox-cdeps-cpu/kaldi/tools/openfst/lib:/share/hhadian/dev/behavox-cdeps-cpu/kaldi/src/lib/

if [ ! -L steps ]; then
    ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
fi

if [ ! -L utils ]; then
    ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
fi

if [ -f $KALDI_ROOT/tools/env.sh ]; then
  . $KALDI_ROOT/tools/env.sh
fi
export PYTHONUNBUFFERED=1
