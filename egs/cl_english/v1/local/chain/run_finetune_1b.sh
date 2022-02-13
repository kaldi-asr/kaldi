#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

# Fine tunes a model on new data. This recipe does not add new layers or freeze the
# weights. Simple fine tuning. The alignments come from the target GMM model
# but the tree is from the source model.
# So nothing in the network is re-initialized.
# By defualt it does not use ivectors.

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
xent_regularize=0.1
chunk_width=140,100,160
nj=30
epochs=1
exp=exp
fpi=1000000
num_leaves=5000
dropout_schedule='0,0@0.20,0.3@0.50,0'
ilr=0.000065
flr=0.0000065
use_ivector=false
final_combine=false
save_interval=10
src_model_has_logsoftmax=true   # dummy variable for consistency with other scripts

gmm=tri3b
lang=data/lang_chain
train_set=train_cv_50k
test_sets=""

# These are for optional full control -- note that augmentation is not done
# in this script.
dir=
ali_dir=
lores_train_data_dir=
train_data_dir=
train_ivector_dir=
tree_dir=

# configs for transfer learning
src_mdl=exp/chain/tdnn1b/final.mdl # Input chain model
src_tree_dir=exp/chain/tree_sp
src_mfcc_config=
src_ivec_extractor_dir=
common_egs_dir=
primary_lr_factor=0.0  # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.

nnet3_affix=
# End configuration section.

echo "$0 $@"  # Print the command line for logging
fullcmd="$0 $@"

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

src_dir=$(dirname $src_mdl)
[ -z $dir ] && dir=$exp/chain/tdnn_ft1b_${affix}
[ -z $train_data_dir ] && train_data_dir=data/${train_set}_sp_hires
[ -z $lores_train_data_dir ] && lores_train_data_dir=data/${train_set}_sp
[ -z $train_ivector_dir ] && train_ivector_dir=$exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
[ -z $ali_dir ] && ali_dir=$exp/${gmm}_ali_${train_set}_sp
[ -z $tree_dir ] && tree_dir=$exp/chain/tree_src${affix:+_$affix}

lang=data/lang_chain
mkdir -p $dir
echo $fullcmd >> $dir/cmd

[ ! -z $common_egs_dir ] && echo $common_egs_dir > $dir/common_egs_dir.txt
[ ! -z $src_tree_dir ] && echo $src_tree_dir > $dir/src_tree_dir.txt

local/chain/run_perturb_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --test-sets "$test_sets"

# Skip stages before 8 because they are for gmm alignemnts etc.
local/chain/run_chain_common.sh \
  --use-ivector $use_ivector \
  --extractor "$src_ivec_extractor_dir" \
  --exp $exp \
  --stage $stage \
  --lores_train_data_dir $lores_train_data_dir \
  --train_data_dir $train_data_dir \
  --gmm $exp/$gmm \
  --ali_lats_dir $ali_dir \
  --lang data/lang_nosp \
  --lang_chain data/lang_chain \
  --tree_dir "" \
  --leaves $num_leaves \
  --test_sets "$test_sets" \
  --nj $nj \
  --nnet3-affix "$nnet3_affix" || exit 1;

src_dir=$(dirname $src_mdl)

if [ ! -e $tree_dir ]; then
  ln -sr $src_tree_dir $tree_dir
fi

if [ $stage -le 16 ]; then
  ln -srf $src_dir/tree $dir/
  ln -srf $src_dir/den.fst $dir/
  ln -srf $src_dir/phone_lm.fst $dir/
  ln -srf $src_dir/normalization.fst $dir/
  ln -srf $src_dir/0.raw $dir/
  ln -srf $src_dir/0.trans_mdl $dir/

  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy $src_mdl $dir/input.raw
fi
train_stage=$(( $train_stage > -3 ? $train_stage : -3 ))

if [ $stage -le 17 ]; then
  echo "$0: generate egs for chain to train new model on rm dataset."
  ivector_dir=
  if $use_ivector; then ivector_dir=$train_ivector_dir ; fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$ivector_dir" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.xent-regularize $xent_regularize \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--max_shuffle_jobs_run 30 \
    --frames-overlap-per-eg 0 --generate_egs_scp true --constrained false" \
    --egs.chunk-width 150 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.frames-per-iter $fpi \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-epochs $epochs \
    --trainer.optimization.num-jobs-initial=4 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=$ilr \
    --trainer.optimization.final-effective-lrate=$flr \
    --trainer.optimization.do-final-combination=$final_combine \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $ali_dir \
    --use-gpu=wait \
    --cleanup.preserve-model-interval=$save_interval \
    --dir $dir || exit 1;
fi
