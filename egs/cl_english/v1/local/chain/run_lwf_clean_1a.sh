#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

# This script fine tunes a model on new data using Learning Without
# Forgetting (LWF) regularization which is a Continual learning (CL) method.
# This recipe does not add new layers or freeze the
# weights. The alignments come from the source model.
# So nothing in the network is re-initialized.
# By defualt it does not use ivectors.
# By default, it uses a variant of LWF called DenLWF as explained in the
# linked paper below. However you may use standard LWF by setting lwf_den_scale
# to "" and setting lwf_scale to non-zero.
# Note that currently this only works with unconstrained egs (so
# you must use "--constrained false").
# See this paper for more info: https://arxiv.org/abs/2110.07055

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
xent_regularize=0.0
chunk_width=140,100,160
nj=30
epochs=1
exp=exp
fpi=1000000

dropout_schedule='0,0@0.20,0.3@0.50,0'
ilr=0.00002
flr=0.000002
use_ivector=false
src_model_has_logsoftmax=false
save_interval=10
chain_tolerance=5
keep_xent=false
remove_egs=true
gmm_ali=true  # set to false if the target train data is too small

lwf_den_scale=0.6  # Set to some value between 0.4 and 1.0, enables DenLWF
lwf_scale=        # Set to some value between 0.7 and 1.3, enables LWF

egs0_dir=
egs_gpu=no
egs_lwf_opts=

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
[ -z $dir ] && dir=$exp/chain/tdnn_lwf1a_${affix}
[ -z $train_data_dir ] && train_data_dir=data/${train_set}_sp_hires
[ -z $lores_train_data_dir ] && lores_train_data_dir=data/${train_set}_sp
[ -z $train_ivector_dir ] && train_ivector_dir=$exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
if $gmm_ali; then
  [ -z $ali_dir ] && ali_dir=$exp/${gmm}_ali_${train_set}_sp
  chain_tolerance=5
  ass=3
  full_gmm_path=$exp/$gmm
else
  [ -z $ali_dir ] && ali_dir=$src_dir/ali_${train_set}_sp
  chain_tolerance=2
  ass=1
  full_gmm_path=""
fi
[ -z $tree_dir ] && tree_dir=$exp/chain/tree_src${affix:+_$affix}
[ -z $egs0_dir ] && egs0_dir=$dir/egs0

lang=data/lang_chain
mkdir -p $dir
echo $fullcmd >> $dir/cmd

if [ ! -e $tree_dir ]; then
  ln -sr $src_tree_dir $tree_dir
fi

if [ -z $lwf_scale ]; then # DenLWF
  egs_lwf_opts="--type=den --clip-threshold=1e-5"
  lwf_scale=0.0
else
  egs_lwf_opts="--type=raw --clip-threshold=1e-5"
  lwf_den_scale=0.0
fi

[ ! -z $common_egs_dir ] && echo $common_egs_dir > $dir/common_egs_dir.txt
[ ! -z $src_tree_dir ] && echo $src_tree_dir > $dir/src_tree_dir.txt
echo "$src_mdl" > $dir/src_model.txt

local/chain/run_perturb_common.sh --stage $stage --nj $nj --train-set $train_set --test-sets "$test_sets"
# Skip stages before 8 because they are for gmm alignemnts etc.
local/chain/run_chain_common.sh \
  --use-ivector $use_ivector \
  --extractor "$src_ivec_extractor_dir" \
  --exp $exp \
  --stage $stage \
  --lores_train_data_dir $lores_train_data_dir \
  --train_data_dir $train_data_dir \
  --gmm $full_gmm_path \
  --ali_lats_dir $ali_dir \
  --lang data/lang_nosp \
  --lang_chain data/lang_chain \
  --tree_dir "" \
  --test_sets "$test_sets" \
  --nj $nj \
  --nnet3-affix "$nnet3_affix" || exit 1;


needs_softmax=true
if $src_model_has_logsoftmax; then
  needs_softmax=false
fi

echo "src model has logsofmax: $src_model_has_logsoftmax"
echo "full post needs sofmax: $needs_softmax"
printf "\n\nAli dir: $ali_dir\n\n"

if [ $stage -le 12 ]; then
  if ! $gmm_ali; then
    echo "$0: Getting alignments using the source chain model...."
    mkdir -p $ali_dir
    echo "$src_dir" >> $ali_dir/source
    ivec_opts=
    if $use_ivector; then
      ivec_opts="--online_ivector_dir $train_ivector_dir"
    fi
    if [ ! -f $ali_dir/ali.1.gz ]; then
      steps/nnet3/align_lats.sh \
        --nj $nj --cmd "$train_cmd" $ivec_opts --acoustic-scale 1.0  \
        --generate-ali-from-lats true \
        --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' $train_data_dir $lang $src_dir $ali_dir
      echo "" >$ali_dir/splice_opts
    fi
  fi
fi

cp -f $src_dir/{tree,den.fst,phone_lm.fst,normalization.fst,0.trans_mdl} $dir/
ln -srf $src_dir/configs $dir/

if [ $stage -le 16 ]; then
  if $src_model_has_logsoftmax; then
    ln -srf $src_dir/0.raw $dir/
    $train_cmd $dir/log/generate_input_mdl.log \
               nnet3-copy $src_mdl $dir/input0.raw
  else  # add a softmax on top
    num_targets=$(tree-info --print-args=false $dir/tree |grep num-pdfs|awk '{print $2}')
    mkdir -p $dir/configs
    cat <<EOF > $dir/configs/final.config
component name=output.log-softmax type=LogSoftmaxComponent dim=$num_targets
component-node name=output.log-softmax component=output.log-softmax input=output.affine
output-node name=output input=output.log-softmax objective=linear
EOF
    $train_cmd $dir/log/generate_input_mdl.log \
               nnet3-copy --edits="remove-output-nodes name=output;" $src_mdl - \| \
               nnet3-init --srand=1 - $dir/configs/final.config $dir/input0.raw  || exit 1;
  fi
  if $keep_xent; then
    cp $dir/input0.raw $dir/input.raw
  else
    nnet3-copy --edits="remove-output-nodes name=output-xent*;remove-orphans;" \
               $dir/input0.raw $dir/input.raw || exit 1;
  fi
fi

if [[ -z $common_egs_dir ]] && [[ $stage -le 17 ]]; then
  . $src_dir/configs/vars
  left_context=$model_left_context
  right_context=$model_right_context
  egs_left_context=$(perl -e "print int($left_context + 3 / 2)") # frame subsampling factor = 3
  egs_right_context=$(perl -e "print int($right_context + 3 / 2)")
  if [ ! -f $egs0_dir/cegs.1.ark ]; then
    steps/nnet3/chain/get_egs.sh \
      --frames-overlap-per-eg 0 --generate-egs-scp true --cmd run.pl \
      --cmvn-opts "--norm-means=false --norm-vars=false" \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial -1 --right-context-final -1 --left-tolerance $chain_tolerance \
      --right-tolerance $chain_tolerance --frame-subsampling-factor 3 --alignment-subsampling-factor $ass \
      --stage 0 --frames-per-iter $fpi --frames-per-eg 150 --srand 0 \
      --max_shuffle_jobs_run 30 --constrained false \
      $train_data_dir $dir $ali_dir $egs0_dir
  fi
  if [ ! -f $dir/egs/cegs.1.ark ]; then
    num_archives=$(cat $egs0_dir/info/num_archives)
    mkdir -p $dir/egs/log
    rm -rf $dir/egs/info || true
    ln -sr $egs0_dir/info $dir/egs/
    test_mode=true # test mode for nnet computation
    $train_cmd JOB=1:$num_archives $dir/egs/log/add_full_post.JOB.log \
               nnet3-chain-add-post-to-egs $egs_lwf_opts \
               --batchnorm-test-mode=$test_mode --dropout-test-mode=$test_mode \
               --use-gpu=$egs_gpu $dir/input.raw $dir/den.fst \
               ark:$egs0_dir/cegs.JOB.ark ark:$dir/egs/cegs.JOB.ark
    $train_cmd $dir/egs/log/add_full_post.train.log \
               nnet3-chain-add-post-to-egs $egs_lwf_opts \
               --batchnorm-test-mode=$test_mode --dropout-test-mode=$test_mode \
               --use-gpu=$egs_gpu $dir/input.raw $dir/den.fst \
               ark:$egs0_dir/train_diagnostic.cegs ark:$dir/egs/train_diagnostic.cegs
    $train_cmd $dir/egs/log/add_full_post.valid.log \
               nnet3-chain-add-post-to-egs $egs_lwf_opts \
               --batchnorm-test-mode=$test_mode --dropout-test-mode=$test_mode \
               --use-gpu=$egs_gpu $dir/input.raw $dir/den.fst \
               ark:$egs0_dir/valid_diagnostic.cegs ark:$dir/egs/valid_diagnostic.cegs
  fi
fi
[ -z "$common_egs_dir" ] && common_egs_dir=$dir/egs


train_stage=$(( $train_stage > -2 ? $train_stage : -2 ))
if [ $stage -le 18 ]; then
  ivector_dir=
  if $use_ivector; then ivector_dir=$train_ivector_dir ; fi
  lwf_opts="--lwf-scale=$lwf_scale --lwf-den-scale=$lwf_den_scale"

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
    --chain.alignment-subsampling-factor $ass \
    --egs.opts "--max-shuffle-jobs-run 30 \
    --frames-overlap-per-eg 0 --generate-egs-scp true --constrained false" \
    --egs.chunk-width 150 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.frames-per-iter $fpi \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.add-option="--optimization.memory-compression-level=1" \
    --trainer.num-epochs $epochs \
    --trainer.optimization.num-jobs-initial=4 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=$ilr \
    --trainer.optimization.final-effective-lrate=$flr \
    --trainer.optimization.do-final-combination=false \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $ali_dir \
    --use-gpu=wait \
    --cleanup.preserve-model-interval=$save_interval \
    --chain-opts="$lwf_opts" \
    --chain.left-tolerance=$chain_tolerance \
    --chain.right-tolerance=$chain_tolerance \
    --dir $dir || exit 1;
fi
