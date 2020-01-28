#!/usr/bin/env bash
# This script contains online decoding using chain + nnet3 setup.
# _6v is as _6h, but moving to a TDNN+ReLU recipe instead of using jesus-layer.
# Otherwise we make everything as similar as possible to 6h.
# The ReLU dimension, at 576, is chosen to make the number of parameters about
# the same as 6h.

# great improvement!
# local/chain/compare_wer.sh 6h 6v
# System                       6h        6v
# WER on train_dev(tg)      15.46     15.00
# WER on train_dev(fg)      14.28     13.91
# WER on eval2000(tg)        17.4      17.2
# WER on eval2000(fg)        15.7      15.7

# the following objf values are computed on the last iter (323), because due to
# a script bug, now fixed, the 'final' ones were not computed in 6v.
# note: in this run the xent learning rate was too slow.
# 323 train prob        -0.129285     -0.120026
# 323 valid prob        -0.151648     -0.140628
# 323 train prob (xent)  -1.4443      -1.5431
# 323 valid prob (xent)  -1.51731     -1.56975


set -e

# configs for 'chain'
affix=
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_6v  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# smoothing options
self_repair_scale=0.00001
# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
relu_dim=576
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1



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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_2y_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
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
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";
  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  fi

  repair_opts=${self_repair_scale:+" --self-repair-scale-nonlinearity $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py \
    $repair_opts \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --tree-dir $treedir \
    $dim_opts \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target $final_layer_normalize_target \
    $dir/configs || exit 1;
fi



if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir exp/chain/tdnn_2y_sp/egs \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1200000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 14 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi

# Results using offline and online decoding
# System                    6v_sp 6v_sp_online 6v_sp_online{per_utt}
# WER on train_dev(tg)      14.68  14.72  15.43
# WER on train_dev(fg)      13.49  13.58  14.18
# WER on eval2000(tg)        17.2  17.3   18.2
# WER on eval2000(fg)        15.7  15.9   16.7

#if [ $stage -le 15 ]; then
#  # If this setup used PLP features, we'd have to give the option --feature-type plp
#  # to the script below.
#  steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
#      data/lang exp/nnet3/extractor "$dir" ${dir}_online || exit 1;
#fi



#if [ $stage -le 16 ]; then
#  iter_opts=
#  if [ ! -z $decode_iter ]; then
#    iter_opts=" --iter $decode_iter "
#  fi
#  for decode_set in train_dev eval2000; do
#      (
#      steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
#          --nj 50 --cmd "$decode_cmd" $iter_opts --config conf/decode_online.config \
#          $graph_dir data/${decode_set}_hires ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
#      if $has_fisher; then
#          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
#            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
#            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
#      fi
#      ) &
#  done
#fi
#
#if [ $stage -le 17 ]; then
#  iter_opts=
#  if [ ! -z $decode_iter ]; then
#    iter_opts=" --iter $decode_iter "
#  fi
#  for decode_set in train_dev eval2000; do
#      (
#      steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --config conf/decode_online.config \
#          --nj 50 --cmd "$decode_cmd" $iter_opts --per-utt true \
#          $graph_dir data/${decode_set}_hires ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}_per_utt || exit 1;
#      if $has_fisher; then
#          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
#            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
#            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg}_per_utt || exit 1;
#      fi
#      ) &
#  done
#fi
#
wait;

exit 0;
