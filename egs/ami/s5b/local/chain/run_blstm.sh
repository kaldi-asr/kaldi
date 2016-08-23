#!/bin/bash

# This is a chain-training script with TDNN neural networks.
# Please see RESULTS_* for examples of command lines invoking this script.


# local/nnet3/run_blstm.sh --stage 8 --use-ihm-ali true --mic sdm1 # rerunning with biphone
# local/nnet3/run_blstm.sh --stage 8 --use-ihm-ali false --mic sdm1

# local/chain/run_blstm.sh --use-ihm-ali true --mic sdm1 --train-set train --gmm tri3 --nnet3-affix "" --stage 12 &

# local/chain/run_blstm.sh --use-ihm-ali true --mic mdm8 --stage 12 &
# local/chain/run_blstm.sh --use-ihm-ali true --mic mdm8 --train-set train --gmm tri3 --nnet3-affix "" --stage 12 &

# local/chain/run_blstm.sh --mic sdm1 --use-ihm-ali true --train-set train_cleaned  --gmm tri3_cleaned&


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=15 # assuming you already ran the tdnn+chain system (local/chain/run_tdnn.sh)
mic=ihm
nj=30
min_seg_len=1.55
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # the gmm for the target data
ihm_gmm=tri3  # the gmm for the IHM system (if --use-ihm-ali true).
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
get_egs_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
blstm_affix=  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

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
                                  --mic $mic \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"

# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --min-seg-len $min_seg_len \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set

if $use_ihm_ali; then
  gmm_dir=exp/ihm/${ihm_gmm}
  ali_dir=exp/${mic}/${ihm_gmm}_ali_${train_set}_sp_comb_ihmdata
  lores_train_data_dir=data/$mic/${train_set}_ihmdata_sp_comb
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}_ihmdata
  lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats_ihmdata
  dir=exp/$mic/chain${nnet3_affix}/blstm${blstm_affix}_sp_bi_ihmali
  # note: the distinction between when we use the 'ihmdata' suffix versus
  # 'ihmali' is pretty arbitrary.
else
  gmm_dir=exp/${mic}/$gmm
  ali_dir=exp/${mic}/${gmm}_ali_${train_set}_sp_comb
  lores_train_data_dir=data/$mic/${train_set}_sp_comb
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}
  lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats
  dir=exp/$mic/chain${nnet3_affix}/blstm${blstm_affix}_sp_bi
fi

train_data_dir=data/$mic/${train_set}_sp_hires_comb
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7


# Note: the first stage of the following script is stage 11.
# runs the common stages in chain models viz., numerator lattice generation,
# and tree generation
local/chain/run_chain_common.sh --stage $stage \
                                --nj $nj \
                                --cmd "$train_cmd" \
                                $lores_train_data_dir $gmm_dir \
                                $ali_dir $lat_dir $tree_dir

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

xent_regularize=0.2
chunk_left_context=40
chunk_right_context=40
frames_per_chunk=150

if [ $stage -le 15 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs";

  steps/nnet3/lstm/make_configs.py  \
    --feat-dir data/$mic/${train_set}_sp_hires_comb \
    --ivector-dir $train_ivector_dir \
    --tree-dir $tree_dir \
    --splice-indexes="-2,-1,0,1,2 0 0" \
    --lstm-delay=" [-3,3] [-3,3] [-3,3] " \
    --xent-regularize $xent_regularize \
    --include-log-softmax false \
    --num-lstm-layers 3 \
    --cell-dim 512 \
    --hidden-dim 512 \
    --recurrent-projection-dim 128 \
    --non-recurrent-projection-dim 128 \
    --label-delay 0 \
    --self-repair-scale-nonlinearity 0.00001 \
    --self-repair-scale-clipgradient 1.0 \
   $dir/configs || exit 1;

fi

if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --chain.left-deriv-truncate 0 \
    --egs.stage $get_egs_stage \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --egs.chunk-left-context 40 \
    --egs.chunk-right-context 40 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 5 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


graph_dir=$dir/graph_${LM}
if [ $stage -le 17 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --left-biphone --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 18 ]; then
  rm $dir/.error 2>/dev/null || true
  extra_left_context=$[$chunk_left_context+10]
  extra_right_context=$[$chunk_right_context+10]
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --extra-left-context $extra_left_context \
          --extra-right-context $extra_right_context \
          --frames-per-chunk $chunk_width \
          --online-ivector-dir exp/$mic/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
