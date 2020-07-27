#!/usr/bin/env bash

# 1c is as 1b but using more 'chain-like' splicing and slightly
# smaller dim.  Not better; maybe slightly worse.

# note: the num-params is almost the same.
# steps/info/nnet3_dir_info.pl exp/nnet3_cleaned/tdnn1{b,c}_sp
# exp/nnet3_cleaned/tdnn1b_sp: num-iters=240 nj=2..12 num-params=10.3M dim=40+100->4187 combine=-0.95->-0.95 loglike:train/valid[159,239,combined]=(-1.01,-0.95,-0.94/-1.18,-1.16,-1.15) accuracy:train/valid[159,239,combined]=(0.71,0.72,0.72/0.67,0.68,0.68)
# exp/nnet3_cleaned/tdnn1c_sp: num-iters=240 nj=2..12 num-params=10.1M dim=40+100->4187 combine=-1.16->-1.15 loglike:train/valid[159,239,combined]=(-1.22,-1.16,-1.15/-1.41,-1.38,-1.38) accuracy:train/valid[159,239,combined]=(0.66,0.67,0.68/0.62,0.63,0.63)

# local/nnet3/compare_wer.sh exp/nnet3_cleaned/tdnn1{b,c}_sp
# System                tdnn1b_sp tdnn1c_sp
# WER on dev(orig)           11.7      11.9
# WER on dev(rescored)       10.9      11.1
# WER on test(orig)          11.7      11.8
# WER on test(rescored)      11.0      11.2
# Final train prob        -0.9416   -1.1505
# Final valid prob        -1.1496   -1.3805
# Final train acc          0.7241    0.6756
# Final valid acc          0.6788    0.6255

#    This is the standard "tdnn" system, built in nnet3; this script
# is the version that's meant to run with data-cleanup, that doesn't
# support parallel alignments.


# steps/info/nnet3_dir_info.pl exp/nnet3_cleaned/tdnn1b_sp
# exp/nnet3_cleaned/tdnn1b_sp: num-iters=240 nj=2..12 num-params=10.3M dim=40+100->4187 combine=-0.95->-0.95 loglike:train/valid[159,239,combined]=(-1.01,-0.95,-0.94/-1.18,-1.16,-1.15) accuracy:train/valid[159,239,combined]=(0.71,0.72,0.72/0.67,0.68,0.68)

# local/nnet3/compare_wer.sh exp/nnet3_cleaned/tdnn1a_sp exp/nnet3_cleaned/tdnn1b_sp
# System                tdnn1a_sp tdnn1b_sp
# WER on dev(orig)           11.9      11.7
# WER on dev(rescored)       11.2      10.9
# WER on test(orig)          11.6      11.7
# WER on test(rescored)      11.0      11.0
# Final train prob        -0.9255   -0.9416
# Final valid prob        -1.1842   -1.1496
# Final train acc          0.7245    0.7241
# Final valid acc          0.6771    0.6788


# by default, with cleanup:
# local/nnet3/run_tdnn.sh

# without cleanup:
# local/nnet3/run_tdnn.sh  --train-set train --gmm tri3 --nnet3-affix "" &


set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
decode_nj=30
min_seg_len=1.55
train_set=train_cleaned
gmm=tri3_cleaned  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for exp dirs, e.g. _cleaned
tdnn_affix=1c  #affix for TDNN directory e.g. "a" or "b", in case we change the configuration.

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
remove_egs=true
srand=0
reporting_email=dpovey@gmail.com
# set common_egs_dir to use previously dumped egs.
common_egs_dir=

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
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"



gmm_dir=exp/${gmm}
graph_dir=$gmm_dir/graph
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
dir=exp/nnet3${nnet3_affix}/tdnn${tdnn_affix}_sp
train_data_dir=data/${train_set}_sp_hires_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
     $graph_dir/HCLG.fst $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=750
  relu-renorm-layer name=tdnn2 dim=750 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=750 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=750 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=750 input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi



if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # note: for TDNNs, looped decoding gives exactly the same results
  # as regular decoding, so there is no point in testing it separately.
  # We use regular decoding because it supports multi-threaded (we just
  # didn't create the binary for that, for looped decoding, so far).
  rm $dir/.error || true 2>/dev/null
  for dset in dev test; do
   (
    steps/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
      ${graph_dir} data/${dset}_hires ${dir}/decode_${dset} || exit 1
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
