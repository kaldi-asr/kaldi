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
resolve_overlaps=false   # Set this to true, if there are overlapping segments
                         # as input and the words in the CTM in the 
                         # overlapping regions must be resolved to one 
                         # of the segments.
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
lang_dir=$(echo "$2" | perl -pe 's/\/$//g')
decode_dir=$3;

##NB: The first ".done" files are used for backward compatibility only
##NB: should be removed in a near future...
if ! $skip_stt ; then
  if  [ ! -f $decode_dir/.score.done ] && [ ! -f $decode_dir/.done.score ]; then
    local/lattice_to_ctm.sh --cmd "$cmd" --word-ins-penalty $wip \
      --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} --resolve-overlaps $resolve_overlaps \
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
  [ ! -f $data_dir/extra_kws_tasks ] && exit 0

  idata=$(basename $data_dir)
  idir=$(dirname $data_dir)

  idataset=${idata%%.*}
  idatatype=${idata#*.}

  if [ "$idata" == "$idataset" ]; then
    syll_data_dir=$idir/${idataset}.syll
    phn_data_dir=$idir/${idataset}.phn
  else
    syll_data_dir=$idir/${idataset}.syll.${idatatype}
    phn_data_dir=$idir/${idataset}.phn.${idatatype}
  fi

  if [ -d ${syll_data_dir} ] && [ ! -f ${decode_dir}/syllabs/.done ] ; then
    local/syllab/lattice_word2syll.sh --cmd "$cmd --mem 8G" \
      $data_dir $lang_dir ${lang_dir}.syll $decode_dir ${decode_dir}/syllabs
    touch ${decode_dir}/syllabs/.done
  fi

  if [ -d ${phn_data_dir} ] && [ ! -f ${decode_dir}/phones/.done ] ; then
    local/syllab/lattice_word2syll.sh --cmd "$cmd --mem 8G" \
      $data_dir $lang_dir ${lang_dir}.phn $decode_dir ${decode_dir}/phones
    touch ${decode_dir}/phones/.done
  fi



  for extraid in `cat $data_dir/extra_kws_tasks | grep -v oov` ; do
    if [ ! -f $decode_dir/.done.kwset.$extraid ] ; then
      local/search/search.sh --cmd "$cmd"  --extraid ${extraid} \
        --max-states ${max_states} --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
        --indices-dir $decode_dir/kws_indices --skip-scoring $skip_scoring \
        $lang_dir $data_dir $decode_dir
      touch $decode_dir/.done.kwset.$extraid
    fi

    if [ -f ${decode_dir}/syllabs/kwset_${extraid}_${min_lmwt}/f4de/metrics.txt ]; then
      touch $decode_dir/syllabs/.done.kwset.$extraid
    fi

    if [ -f ${decode_dir}/phones/kwset_${extraid}_${min_lmwt}/f4de/metrics.txt ]; then
      touch $decode_dir/phones/.done.kwset.$extraid
    fi

    if [ -f ${decode_dir}/syllabs/.done ] && [ ! -f $decode_dir/syllabs/.done.kwset.$extraid ] ; then
      local/search/search.sh --cmd "$cmd"  --extraid ${extraid} --model $decode_dir/../final.mdl\
        --max-states ${max_states} --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
        --indices-dir $decode_dir/syllabs/kws_indices --skip-scoring $skip_scoring \
        ${lang_dir}.syll $syll_data_dir $decode_dir/syllabs
      touch $decode_dir/syllabs/.done.kwset.$extraid
    fi


    if [ -f ${decode_dir}/phones/.done ] && [ ! -f $decode_dir/phones/.done.kwset.$extraid ] ; then
      local/search/search.sh --cmd "$cmd"  --extraid ${extraid} --model $decode_dir/../final.mdl\
          --max-states ${max_states} --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
          --indices-dir $decode_dir/phones/kws_indices --skip-scoring $skip_scoring \
          ${lang_dir}.phn $phn_data_dir $decode_dir/phones
      touch $decode_dir/phones/.done.kwset.$extraid
    fi
  done
fi
