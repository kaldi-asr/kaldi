#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail
set -u

stage=-1
decode_stage=1

mic=ihm
use_ihm_ali=false
exp_name=tdnn

nj=20

cleanup_affix=
graph_dir=

decode_set=dev
decode_suffix=

extractor=
use_ivectors=true
use_offline_ivectors=false
frames_per_chunk=50

scoring_opts=

. path.sh
. cmd.sh

. parse_options.sh

new_mic=$mic
if [ $use_ihm_ali == "true" ]; then
  new_mic=${mic}_cleanali
fi

dir=exp/$new_mic/chain${cleanup_affix:+_$cleanup_affix}/${exp_name}

if [ $stage -le -1 ]; then
  mfccdir=mfcc_${mic}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc.conf \
    --cmd "$train_cmd" data/$mic/${decode_set} exp/make_${mic}/$decode_set $mfccdir || exit 1;
  
  steps/compute_cmvn_stats.sh data/$mic/${decode_set} exp/make_${mic}/$mic/$decode_set $mfccdir || exit 1;

  utils/fix_data_dir.sh data/$mic/${decode_set}
fi

if [ $stage -le 0 ]; then
  mfccdir=mfcc_${mic}_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  utils/copy_data_dir.sh data/$mic/$decode_set data/$mic/${decode_set}_hires

  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/$mic/${decode_set}_hires exp/make_${mic}_hires/$decode_set $mfccdir || exit 1;
  
  steps/compute_cmvn_stats.sh data/$mic/${decode_set}_hires exp/make_${mic}_hires/$mic/$decode_set $mfccdir || exit 1;

  utils/fix_data_dir.sh data/$mic/${decode_set}_hires
fi

if $use_ivectors && [ $stage -le 1 ]; then
  if [ -z "$extractor" ]; then
    echo "--extractor must be supplied when using ivectors"
    exit 1
  fi

  if $use_offline_ivectors; then
    steps/online/nnet2/extract_ivectors.sh \
      --cmd "$train_cmd" --nj 8 \
      data/$mic/${decode_set}_hires data/lang $extractor \
      exp/$mic/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_offline_${decode_set} || exit 1
  else
    steps/online/nnet2/extract_ivectors_online.sh \
      --cmd "$train_cmd" --nj 8 \
      data/$mic/${decode_set}_hires $extractor \
      exp/$mic/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_${decode_set} || exit 1
  fi
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

if [ -z "$graph_dir" ]; then
  graph_dir=$dir/graph_${LM}
  if [ $stage -le 2 ]; then
    # Note: it might appear that this $lang directory is mismatched, and it is as
    # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
    # the lang directory.
    utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
  fi
fi

nj=`cat data/$mic/${decode_set}/utt2spk|cut -d' ' -f2|sort -u|wc -l`

if [ $nj -gt 50 ]; then
  nj=50
fi

if [ "$frames_per_chunk" -ne 50 ]; then
  decode_suffix=${decode_suffix}_cs${frames_per_chunk}
fi

if [ $stage -le 3 ]; then
  ivector_opts=
  if $use_ivectors; then
    if $use_offline_ivectors; then
      ivector_opts="--online-ivector-dir exp/$mic/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_offline_${decode_set}"
      decode_suffix=${decode_suffix}_offline
    else
      ivector_opts="--online-ivector-dir exp/$mic/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_${decode_set}"
    fi
  fi
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --stage $decode_stage --frames-per-chunk $frames_per_chunk \
    --nj $nj --cmd "$decode_cmd" $ivector_opts \
    --scoring-opts "--min-lmwt 5 --decode-mbr false $scoring_opts" \
    $graph_dir data/$mic/${decode_set}_hires $dir/decode${decode_suffix}_${decode_set} || exit 1;
fi
