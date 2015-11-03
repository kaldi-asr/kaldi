#!/bin/bash


# This version of the TDNN system is being built to have a similar configuration
# to the one in local/online/run_nnet2.sh, for better comparability.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
dir=exp/nnet3/nnet_tdnn_c
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "-1,0,1  -2,1  -4,2 0" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train_si284 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --io-opts "--max-jobs-run 12" \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 250 \
    data/train_si284_hires data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_test_$year \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi

# The following results compare this nnet3 decode with the matched nnet2 baseline.
#
#b06:s5: cat exp/nnet3/nnet_tdnn_c/decode_*/scoring_kaldi/best_wer 
#%WER 6.40 [ 527 / 8234, 59 ins, 71 del, 397 sub ] exp/nnet3/nnet_tdnn_c/decode_bd_tgpr_dev93/wer_10_0.0
#%WER 3.54 [ 200 / 5643, 18 ins, 17 del, 165 sub ] exp/nnet3/nnet_tdnn_c/decode_bd_tgpr_eval92/wer_10_1.0
#%WER 9.11 [ 750 / 8234, 140 ins, 83 del, 527 sub ] exp/nnet3/nnet_tdnn_c/decode_tgpr_dev93/wer_10_1.0
#%WER 6.22 [ 351 / 5643, 85 ins, 15 del, 251 sub ] exp/nnet3/nnet_tdnn_c/decode_tgpr_eval92/wer_10_1.0
#b06:s5: 
#b06:s5: cat exp/nnet2_online/nnet_ms_a/decode_*/scoring_kaldi/best_wer 
#%WER 6.62 [ 545 / 8234, 56 ins, 79 del, 410 sub ] exp/nnet2_online/nnet_ms_a/decode_bd_tgpr_dev93/wer_13_0.0
#%WER 3.70 [ 209 / 5643, 25 ins, 18 del, 166 sub ] exp/nnet2_online/nnet_ms_a/decode_bd_tgpr_eval92/wer_13_0.5
#%WER 9.33 [ 768 / 8234, 157 ins, 73 del, 538 sub ] exp/nnet2_online/nnet_ms_a/decode_tgpr_dev93/wer_11_0.5
#%WER 6.11 [ 345 / 5643, 92 ins, 14 del, 239 sub ] exp/nnet2_online/nnet_ms_a/decode_tgpr_eval92/wer_10_1.0
