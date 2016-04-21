#!/bin/bash

export KALDI_ROOT=`pwd`/../../..
. $KALDI_ROOT/tools/config/common_path.sh

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH

. $KALDI_ROOT/tools/env.sh
export LC_ALL=C

