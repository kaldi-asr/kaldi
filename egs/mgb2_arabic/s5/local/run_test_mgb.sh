#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

set -euo pipefail

. path.sh
. cmd.sh

stage=100
nj=40

seconds_per_spk_max=30

gmm_dir=
chain_dir=exp/chain_mer80/tdnn_lstm1a_sp_bi

lang=data/lang_1200h_test
lang_rescore=data/lang_test_fg

extractor=exp/nnet3_mer80/extractor
extra_left_context=50 
extra_right_context=0
frames_per_chunk=150

. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 [options] <data>"
  echo " e.g.: $0 data/eval.uem"
  exit 1
fi

data=$1

if [ ! -z "$seconds_per_spk_max" ]; then
  utils/data/modify_speaker_info.sh --seconds-per-spk-max $seconds_per_spk_max $data ${data}_spk30sec
  data=${data}_spk30sec
fi

data_id=`basename $data`
lang_id=$(echo $lang | perl -ne 'm:.+/lang(_\S+)_test:; print $1;')

if [ ! -z "$gmm_dir" ]; then
  if [ ! -s $data/feats.scp ]; then
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj $data
    steps/compute_cmvn_stats.sh ${data}
  fi

  if [ ! -f $gmm_dir/graph${lang_id}/HCLG.fst ]; then
    utils/mkgraph.sh $lang $gmm_dir $gmm_dir/graph${lang_id}
  fi

  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
    --config conf/decode.config \
    $gmm_dir $data $gmm_dir/decode${lang_id}_${data_id}
  
  if [ ! -z "$lang_rescore" ]; then
    rescore_id=$(echo $lang_rescore/ | perl -ne 'm:.+test([^/]+)/:; print $1;')
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      --config conf/decode.config \
      $lang $lang_rescore $data \
      $gmm_dir/decode${lang_id}_${data_id} \
      $gmm_dir/decode${lang_id}_${data_id}${rescore_id}
  fi
fi

if [ ! -s ${data}_hires/feats.scp ]; then
  utils/copy_data_dir.sh $data ${data}_hires
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj ${data}_hires
  steps/compute_cmvn_stats.sh ${data}_hires
fi

ivector_dir=`dirname $extractor`/ivectors_${data_id}_hires

if [ $stage -le 6 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${data}_hires $extractor $ivector_dir
fi
  
if [ ! -z "$chain_dir" ]; then
  if [ $stage -le 7 ]; then
    if [ ! -f $chain_dir/graph${lang_id}/HCLG.fst ]; then
      utils/mkgraph.sh --left-biphone --self-loop-scale 1.0 \
        $lang $chain_dir $chain_dir/graph${lang_id}
    fi
  fi
 
  if [ $stage -le 8 ]; then
    steps/nnet3/decode.sh --num-threads 4 --nj $nj --cmd "$decode_cmd" \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --extra-left-context $extra_left_context  \
      --extra-right-context $extra_right_context  \
      --frames-per-chunk "$frames_per_chunk" \
      --online-ivector-dir $ivector_dir \
      --scoring-opts "--min-lmwt 5 --max-lmwt 15" \
      $chain_dir/graph${lang_id} ${data}_hires $chain_dir/decode${lang_id}_${data_id}
  fi

  if [ ! -z "$lang_rescore" ]; then
    rescore_id=$(echo $lang_rescore/ | perl -ne 'm:.+test([^/]+)/:; print $1;')
    if [ $stage -le 9 ]; then
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        --scoring-opts "--min-lmwt 5 --max-lmwt 15" \
        $lang $lang_rescore ${data}_hires \
        $chain_dir/decode${lang_id}_${data_id} \
        $chain_dir/decode${lang_id}_${data_id}${rescore_id}
    fi
  fi
fi
