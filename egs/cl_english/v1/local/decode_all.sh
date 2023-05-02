#!/usr/bin/env bash

# decode all test sets with all graphs


stage=0
dir=exp/chain/tdnn1a
nj=30
graphs="exp/chain/tree_sp/graph_tg"
ivector_base=exp/nnet3
test_sets="test_ldc"
iter=final
decode_opts=""   # will be set to a default if empty
suffix=""
overwrite=false

echo "$0 $@"  # Print the command line for logging
fullcmd="$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

set -e
echo $dir
echo "`date`: $fullcmd" >> $dir/log.decodes.txt
[ -z "$decode_opts" ] && decode_opts="--acwt 1.0 --post-decode-acwt 10.0"

if [[ $stage -le 0 ]] && [[ ! -z $ivector_base ]]; then
  for data in $test_sets; do
    ivector_dir=$ivector_base/ivectors_${data}_hires
    if [ ! -f $ivector_dir/ivector_online.scp ]; then
      steps/online/nnet2/extract_ivectors_online.sh \
        --nj $nj data/${data}_hires $ivector_base/extractor $ivector_dir
    fi
  done
fi

if [ $stage -le 1 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    ivector_dir=$ivector_base/ivectors_${data}_hires
    ivector_opts=
    if [ ! -z $ivector_base ]; then
      ivector_opts="--online-ivector-dir $ivector_dir"
    fi
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    if [ $nspk -gt $nj ]; then
      nspk=$nj
    fi
    for graph in $graphs; do
      graph_suffix=`echo $(basename $graph) | awk -F'_' '{print $NF}'`
      decode_dir=${dir}/decode_${graph_suffix}${suffix}_${data}_iter$iter
      if [[ "$overwrite" == false ]] && [[ -e $decode_dir/scoring_kaldi/best_wer ]]; then
        echo "$0: Skipping $decode_dir as it already exits."
      else
        steps/nnet3/decode.sh --iter $iter \
            $decode_opts $ivector_opts \
            --frames-per-chunk $frames_per_chunk \
            --nj $nspk --cmd "$decode_cmd"  --num-threads 2 \
            $graph data/${data}_hires $decode_dir || exit 1
      fi
      echo ""
      cat $decode_dir/scoring_kaldi/best_wer
      echo ""
    done
  done
fi
