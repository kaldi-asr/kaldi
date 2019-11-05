#! /usr/bin/env bash
#
# Environment variables to use by speech_recognition repo

export KALDI_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

export PATH=$KALDI_ROOT/egs/wsj/s5/utils/:$KALDI_ROOT/tools/openfst/bin${PATH:+:${PATH}}

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

# Make sure that MITLM shared libs are found by the dynamic linker/loader
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tools/mitlm-svn/lib
