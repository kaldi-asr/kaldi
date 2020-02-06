#!/usr/bin/env bash
# Copyright 2013  Johns Hopkins University (authors: Yenda Trmal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

#Simple BABEL-only script to be run on generated lattices (to produce the
#files for scoring and for NIST submission

set -e
set -o pipefail
set -u

#Begin options
min_lmwt=8
max_lmwt=12
cer=0
skip_kws=false
skip_stt=false
skip_scoring=false
extra_kws=false
cmd=run.pl
max_states=150000
wip=0.5 #Word insertion penalty
#End of options

if [ $(basename $0) == score.sh ]; then
  skip_kws=true
fi

echo $0 "$@"
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <decode-dir>"
  echo " e.g.: $0 data/dev10h data/lang exp/tri6/decode_dev10h"
  exit 1;
fi

data_dir=$1;
lang_dir=$2;
decode_dir=$3;

##NB: The first ".done" files are used for backward compatibility only
##NB: should be removed in a near future...
if ! $skip_stt ; then
  if  [ ! -f $decode_dir/.score.done ] && [ ! -f $decode_dir/.done.score ]; then
    local/lattice_to_ctm.sh --cmd "$cmd" --word-ins-penalty $wip \
      --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
      $data_dir $lang_dir $decode_dir

    if ! $skip_scoring ; then
      local/score_stm.sh --cmd "$cmd"  --cer $cer \
        --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt}\
        $data_dir $lang_dir $decode_dir
    fi
    touch $decode_dir/.done.score
  fi
fi

if ! $skip_kws ; then
  if [ ! -f $decode_dir/.kws.done ] && [ ! -f $decode_dir/.done.kws ]; then
    local/kws_search.sh --cmd "$cmd" --max-states ${max_states} \
      --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} --skip-scoring $skip_scoring\
      --indices-dir $decode_dir/kws_indices $lang_dir $data_dir $decode_dir
    touch $decode_dir/.done.kws
  fi
  if $extra_kws && [ -f $data_dir/extra_kws_tasks ]; then
    for extraid in `cat $data_dir/extra_kws_tasks` ; do
      [ -f $decode_dir/.done.kws.$extraid ] && continue;
      local/kws_search.sh --cmd "$cmd" --extraid $extraid  \
        --max-states ${max_states} --min-lmwt ${min_lmwt} --skip-scoring $skip_scoring\
         --max-lmwt ${max_lmwt} --indices-dir $decode_dir/kws_indices \
        $lang_dir $data_dir $decode_dir
      touch $decode_dir/.done.kws.$extraid
    done
  fi
fi
