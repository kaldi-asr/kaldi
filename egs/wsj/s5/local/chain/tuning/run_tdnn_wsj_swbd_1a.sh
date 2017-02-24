#!/bin/bash
# wsj_swbd_1a is as wsj_swbd but it has lower number of final jobs and max-change
# and learning rate in fine tuning stage.
# This script used for transfer learning to transfer information from swbd model to wsj.
# it uses chain training to train primary swbd and secondary wsj models.
# it is based on adding a 64 dim lowrank module in the xent branch.
#System                   tdnn_7h    tdnn_7l
#WER on train_dev(tg)     13.84      13.83
#WER on train_dev(fg)     12.84      12.88
#WER on eval2000(tg)      16.5       16.4
#WER on eval2000(fg)      14.8       14.7
#Final train prob         -0.089     -0.090
#Final valid prob         -0.113     -0.116
#Final train prob (xent)  -1.25      -1.38
#Final valid prob (xent)  -1.36      -1.48
#Time consuming one iter  53.56s     48.18s
#Time reduction percent   10.1%
set -e

# configs for 'chain'
affix=
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_swbd_wsj  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# training options
num_epochs=4
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
minibatch_size=128
frames_per_eg=150
remove_egs=false
srcdir=../../swbd/s5c
common_egs_dir_stage1=$srcdir/exp/chain/tdnn_7i_sp/egs
common_egs_dir_stage2=exp/chain/tdnn_swbd_wsj/stage2/egs
xent_regularize=0.1
src_mdl=$srcdir/exp/chain/tdnn_7h_sp/final.mdl

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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=_hires_downsampled
train_set=train_si284
ali_dir=exp/tri4b_ali_si284
lat_dir=exp/tri4b_lats_si284
tree_dir=exp/chain/tri5_7d_tree
lang=data/lang_chain
train_data_dir=data/${train_set}${suffix}

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mfcc-config conf/mfcc_hires_downsampled.conf \
                                  --suffix $suffix \
                                  --extractor $srcdir/exp/nnet3/extractor || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4b_ali_si284/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${train_set} \
    data/lang_nosp exp/tri4b exp/tri4b_lats_si284
  rm exp/tri4b_lats_si284/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  #if [ -d $lang ]; then
  if false; then
    if [ $lang/L.fst -nt data/lang_nosp/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    rm -rf $lang
    cp -r data/lang_nosp $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 4000 data/${train_set} $lang $ali_dir $tree_dir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser for";
  echo "primary and secondary networks.";

  num_targets=$(tree-info $srcdir/exp/chain/tri5_7d_tree_sp/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  mkdir -p $dir/stage1
  mkdir -p $dir/stage1/configs
  cat <<EOF > $dir/stage1/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/stage1/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=625
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=625
  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain input=tdnn7 dim=625 target-rms=0.5
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
  relu-renorm-layer name=prefinal-xent input=tdnn7 dim=625 target-rms=0.5
  relu-renorm-layer name=prefinal-lowrank-xent input=prefinal-xent dim=64 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/stage1/configs/network.xconfig --config-dir $dir/stage1/configs/

  num_targets_stage2=$(tree-info exp/chain/tri5_7d_tree/tree |grep num-pdfs|awk '{print $2}')
  dim2=450
  mkdir -p $dir/stage2
  mkdir -p $dir/stage2/configs
  cat <<EOF > $dir/stage2/configs/network.xconfig
  relu-renorm-layer name=tdnn5-stage2 input=Append(tdnn4@-3,tdnn4@0,tdnn4@3) dim=$dim2
  relu-renorm-layer name=tdnn6-stage2 input=Append(-6,-3,0) dim=$dim2
  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain-stage2 input=tdnn6-stage2 dim=$dim2 target-rms=0.5
  output-layer name=output-stage2 include-log-softmax=false dim=$num_targets_stage2 max-change=1.5
  relu-renorm-layer name=prefinal-xent-stage2 input=tdnn6-stage2 dim=$dim2 target-rms=0.5
  relu-renorm-layer name=prefinal-lowrank-xent-stage2 input=prefinal-xent-stage2 dim=64 target-rms=0.5
  output-layer name=output-xent-stage2 dim=$num_targets_stage2 learning-rate-factor=$learning_rate_factor max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --aux-xconfig-file $dir/stage1/configs/network.xconfig \
    --xconfig-file $dir/stage2/configs/network.xconfig \
    --config-dir $dir/stage2/configs/
  # edits.config contains edits required for different stage of training.
  cat <<EOF > $dir/stage2/configs/edits.config
  remove-output-nodes name=output
  remove-output-nodes name=output-xent
  rename-node old-name=output-stage2 new-name=output
  rename-node old-name=output-xent-stage2 new-name=output-xent
  remove-orphans
EOF
cp -r $dir/stage1/configs/vars $dir/stage2/configs/.

fi
if [ $stage -le 13 ] && [ ! -f $src_mdl ]; then
  echo "$0: generates egs for 1st dataset and training 1st model using larger dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/stage1/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $srcdir/exp/nnet3/ivectors_train_nodup_sp \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir_stage1" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $srcdir/data/train_nodup_sp_hires \
    --tree-dir $srcdir/exp/chain/tri5_7d_tree_sp \
    --lat-dir $srcdir/exp/tri4_lats_nodup_sp \
    --dir $dir/stage1  || exit 1;

fi

if [ $stage -le 14 ]; then
  echo "$0: generate egs for chain training for 2nd dataset and re-train the model on smaller dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/stage2/egs/storage
  fi
  echo "$0: set the learning-rate-factor for initial network to be zero."
  nnet3-am-copy --raw=true --edits='set-learning-rate-factor name=* learning-rate-factor=0.0' \
    $src_mdl $dir/stage2/init.raw || exit 1;
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --init-raw-model $dir/stage2/init.raw \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir_stage2" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4.0 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.005 \
    --trainer.optimization.final-effective-lrate 0.0005 \
    --trainer.max-param-change 2.0 \
    --trainer.add-layers-period 1 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir/stage2
fi

if [ $stage -le 15 ]; then
  mkdir -p $dir/stage3
  echo "$0: re-train the whole network with smaller learning rate using smaller dataset."
  nnet3-am-copy --edits='set-learning-rate-factor name=* learning-rate-factor=1.0;' \
    $dir/stage2/final.mdl $dir/stage3/0.mdl || exit 1;
  train_stage_s3=0
  if [ $train_stage -gt $train_stage_s3 ]; then
    stage=$train_stage_s3
  fi
  cp -r $dir/stage2/configs $dir/stage3/.
  cp $dir/stage2/den.fst $dir/stage3/.

  steps/nnet3/chain/train.py --stage $train_stage_s3 \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir_stage2" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 1.0 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.0005 \
    --trainer.optimization.final-effective-lrate 0.00005 \
    --trainer.max-param-change 1.0 \
    --trainer.add-layers-period 1 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir/stage3 || exit 1;
fi
if [ $stage -le 16 ]; then
  for dset in $dir/stage2 $dir/stage3;do
    for lm_suffix in tgpr bd_tgpr; do
      utils/mkgraph.sh --self-loop-scale 1.0 data/lang_nosp_test_${lm_suffix} $dset $dset/graph_${lm_suffix}
    done
  done
fi

if [ $stage -le 17 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for dset in $dir/stage2 $dir/stage3;do
    for lm_suffix in tgpr bd_tgpr; do
      # use already-built graphs.
      #for year in eval92 dev93; do
      for year in dev93; do
        steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 \
            --online-ivector-dir exp/nnet3/ivectors_test_$year \
           $dset/graph_${lm_suffix} data/test_${year}${suffix} $dset/decode_${lm_suffix}_${year} || exit 1;
      done
    done
  done
fi
wait;
exit 0;
# results for stage2
%WER 6.16 [ 507 / 8234, 49 ins, 86 del, 372 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_bd_tgpr_dev93/wer_10_0.0
%WER 3.70 [ 209 / 5643, 23 ins, 18 del, 168 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_bd_tgpr_eval92/wer_10_0.5
%WER 8.26 [ 680 / 8234, 110 ins, 93 del, 477 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_tgpr_dev93/wer_9_0.5
%WER 5.88 [ 332 / 5643, 58 ins, 45 del, 229 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_tgpr_eval92/wer_11_0.5
# results for stage3
%WER 6.24 [ 514 / 8234, 58 ins, 91 del, 365 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_bd_tgpr_dev93/wer_8_0.0
%WER 3.62 [ 204 / 5643, 21 ins, 15 del, 168 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_bd_tgpr_eval92/wer_8_0.5
%WER 8.36 [ 688 / 8234, 98 ins, 138 del, 452 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_tgpr_dev93/wer_8_0.0
%WER 5.65 [ 319 / 5643, 51 ins, 59 del, 209 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_tgpr_eval92/wer_10_0.0
# result for stage 3 f-per-iter 450000, epoch=1 max-change=1.0
%WER 6.16 [ 507 / 8234, 49 ins, 86 del, 372 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_bd_tgpr_dev93/wer_10_0.0
%WER 3.70 [ 209 / 5643, 23 ins, 18 del, 168 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_bd_tgpr_eval92/wer_10_0.5
%WER 8.26 [ 680 / 8234, 110 ins, 93 del, 477 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_tgpr_dev93/wer_9_0.5
%WER 5.88 [ 332 / 5643, 58 ins, 45 del, 229 sub ] exp/chain/tdnn_swbd_wsj/stage2/decode_tgpr_eval92/wer_11_0.5

# results for stage3 f-per-iter 1500000, epoch=1 max-change=0.75
%WER 6.32 [ 520 / 8234, 50 ins, 105 del, 365 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_bd_tgpr_dev93/wer_8_0.0
%WER 3.58 [ 202 / 5643, 23 ins, 17 del, 162 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_bd_tgpr_eval92/wer_9_0.5
%WER 8.08 [ 665 / 8234, 68 ins, 184 del, 413 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_tgpr_dev93/wer_9_0.0
%WER 5.60 [ 316 / 5643, 49 ins, 62 del, 205 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_tgpr_eval92/wer_9_1.0

# results for stage3 f-per-iter 1500000, epoch=1 max-change=1.0 init-final lr=0.005,0.0005 init-final jobs=2,2
%WER 6.23 [ 513 / 8234, 46 ins, 99 del, 368 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_bd_tgpr_dev93/wer_8_0.0
%WER 3.40 [ 192 / 5643, 18 ins, 24 del, 150 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_bd_tgpr_eval92/wer_10_0.5
%WER 8.12 [ 669 / 8234, 69 ins, 185 del, 415 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_tgpr_dev93/wer_9_0.0
%WER 5.53 [ 312 / 5643, 52 ins, 53 del, 207 sub ] exp/chain/tdnn_swbd_wsj/stage3/decode_tgpr_eval92/wer_9_0.5
