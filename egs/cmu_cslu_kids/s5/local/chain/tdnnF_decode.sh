#! /bin/bash

# Copyright Johns Hopkins University
#   2019 Fei Wu

# Decode on new data set using trained model. 
# The data directory should be prepared in kaldi style.
# Usage:
#     ./local/chain/tdnnF_decode.sh --data_src <prepared_data_dir> 

set -euo pipefail
echo "$0 $@"

stage=0
decode_nj=10
data_src=
affix=
tree_affix=
nnet3_affix=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
data_name=$(basename $data_src)
data_hires="data/${data_name}_hires"
ivect_dir=exp/nnet3${nnet3_affix}/ivector_$data_name
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp

mfcc=mfcc_hires_$data_name
chunk_width=140,100,160
reporting_email=

if [ $stage -le 0 ]; then 
    rm -rf $data_hires
    cp -r $data_src $data_hires
fi
# High resolution mfcc
if [ $stage -le 1 ]; then 
    steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" $data_hires \
        exp/$data_name/make_feat_hires  $mfcc|| exit 1;
    steps/compute_cmvn_stats.sh $data_hires || exit 1;
    utils/fix_data_dir.sh $data_hires || exit 1;
fi

# Extract i-vector
if [ $stage -le 2 ]; then 
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
        $data_hires exp/nnet3"$affix"/extractor $ivect_dir
fi

if [ $stage -le 3 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

    (
      nspk=$(wc -l <$data_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir $ivect_dir \
          $tree_dir/graph_tgsmall $data_hires ${dir}/decode_tgsmall_$data_name || exit 1
      
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       $data_hires ${dir}/decode_{tgsmall,tglarge}_$data_name || exit 1
    ) || touch $dir/.error &

  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

