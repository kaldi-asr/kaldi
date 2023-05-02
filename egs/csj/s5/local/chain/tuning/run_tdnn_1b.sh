#!/usr/bin/env bash

# This is a spec augmentation experiment.

# steps/info/chain_dir_info.pl exp/chain/tdnn1b
# exp/chain/tdnn1b: num-iters=2610 nj=2..2 num-params=13.6M dim=40+100->3920 combine=-0.075->-0.075 (over 23) xent:train/valid[1737,2609,final]=(-0.949,-0.924,-0.922/-0.950,-0.935,-0.932) logprob:train/valid[1737,2609,final]=(-0.066,-0.063,-0.063/-0.072,-0.071,-0.071)

# local/chain/compare_wer.sh --online exp/chain/tdnn1b
# System                        tdnn1a    tdnn1b
# WER eval1                      10.30     10.05
#          [online:]             10.30     10.05
# WER eval2                       8.59      8.09
#          [online:]              8.56      8.07
# WER eval3                       9.90      9.33
#          [online:]              9.90      9.31
# Final train prob             -0.0532   -0.0628
# Final valid prob             -0.0776   -0.0708
# Final train prob (xent)      -0.8289   -0.9225
# Final valid prob (xent)      -0.9539   -0.9319

set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_nodup
dev_set=
test_sets="eval1 eval2 eval3"
gmm=tri4
nnet3_affix=

# The rest are configs specific to this script. Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1b  # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
decode_iter=
num_epochs=10
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=2
minibatch_size=128,64
frames_per_eg=150,140,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $*"  # Print the command line for logging

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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
  $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0; expected file $f to exist" && exit 1;
done

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 ${lores_train_data_dir} $lang $ali_dir $tree_dir
fi


if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat include-in-init=true
  batchnorm-component name=batchnorm0 input=idct include-in-init=true
  spec-augment-layer name=spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20 include-in-init=true
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn5 dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn9 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn9 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/csj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_csj_tg $dir $dir/graph_csj_tg

  for decode_set in $test_sets; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 10 \
      --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
      $dir/graph_csj_tg data/${decode_set}_hires $dir/decode_${decode_set}
  done

  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf $lang exp/nnet3${nnet3_affix}/extractor \
    $dir ${dir}_online

  for decode_set in $test_sets; do
    steps/online/nnet3/decode.sh --nj 10 --cmd "$decode_cmd" \
      --acwt 1.0 --post-decode-acwt 10.0 \
      $dir/graph_csj_tg data/${decode_set}_hires ${dir}_online/decode_${decode_set}
  done
fi
exit 0;
