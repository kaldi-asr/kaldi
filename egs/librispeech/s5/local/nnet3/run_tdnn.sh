#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# without cleanup:
# local/nnet3/run_tdnn.sh  --train-set train960 --gmm tri6b --nnet3-affix "" &


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=30
train_set=train_960_cleaned
gmm=tri6b_cleaned  # this is the source gmm-dir for the data-type of interest; it
                   # should have alignments for the specified training data.
nnet3_affix=_cleaned

# Options which are not passed through to run_ivector_common.sh
affix=
train_stage=-10
common_egs_dir=
reporting_email=
remove_egs=true

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;


gmm_dir=exp/${gmm}
graph_dir=$gmm_dir/graph_tgsmall
ali_dir=exp/${gmm}_ali_${train_set}_sp
dir=exp/nnet3${nnet3_affix}/tdnn${affix:+_$affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires


for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
     $graph_dir/HCLG.fst $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir $train_data_dir \
    --ivector-dir $train_ivector_dir \
    --ali-dir $ali_dir \
    --relu-dim 1280 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0"  \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi



if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 100 \
    --feat-dir=$train_data_dir \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

if [ $stage -le 13 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  rm $dir/.error 2>/dev/null || true
  for test in test_clean test_other dev_clean dev_other; do
    (
    steps/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${test}_hires \
      ${graph_dir} data/${test}_hires $dir/decode_${test}_tgsmall || exit 1
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/${test}_hires $dir/decode_${test}_{tgsmall,tgmed}  || exit 1
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/${test}_hires $dir/decode_${test}_{tgsmall,tglarge} || exit 1
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/${test}_hires $dir/decode_${test}_{tgsmall,fglarge} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
