#!/bin/bash

# _2b is as _y but --frames-overlap-per-eg 75 (was 30 before).  This is not very
# efficient in terms of disk space but I want to see the effect on results.

# In terms of the objf, the training is a lot better, -0.0879->-0.0779, and validation is
#  slightly better: -0.126 -> -0.123.
# But the WERs are 0.3 worse across the board: on train_dev, with tg 18.04->18.15, with fg
#   16.57->16.83; on all of eval2000, with tg 13.2->13.7, and with fg 11.7->12.0.
# I'm a little at a loss how to interpret these.
#   Note: I decode an earlier iter (300) but the results were not much better: final->300,
#   13.7->13.7 on all of eval2000 with tg, and 18.15->18.10 on all of train_dev with tg.

# _y is as _s but trying --apply-deriv-weights false. (note: in the
# interim, the script was changed so the train and valid probs have --pdf-boundary-penalty 0
# and are no longer comparable with the ones in _s.
#
#   Compared to s, the results are improved: on train_dev, 18.45->18.04 with tg
# and 16.96->16.57 with fg; on all of eval2000, 20.1->19.8 with tg and 18.0 to
# 17.9 with fg.
#
#
#  I recomputed the train and valid probs using the .486 model and no --pdf-boundary-penalty option, to
# be able to compre with the _s ones.  In _s the (train,valid) probs at iter 485 were (-0.0691, -0.0997),
# in _y the (train,valid) probs at iter 486 were (-0.0655,-0.0998).  So better on train, essentially
# the same on valid.  It makes sense it would be better on train, since its overtraining is more
# closely aligned with the distribution of training segments on which we compute the objf-- also because
# we've simply trained more, i.e. equivalent to slightly more epochs.


# _s is as _q but setting pdf-boundary-penalty to 0.0
# This is helpful: 19.8->18.0 after fg rescoring on all of eval2000,
# and 18.07 -> 16.96 on train_dev, after fg rescoring.

# _q is as _p except making the same change as from n->o, which
# reduces the parameters to try to reduce over-training.  We reduce
# relu-dim from 1024 to 850, and target num-states from 12k to 9k,
# and modify the splicing setup.
# note: I don't rerun the tree-building, I just use the '5o' treedir.

# _p is as _m except with a code change in which we switch to a different, more
# exact mechanism to deal with the edges of the egs, and correspondingly
# different script options... we now dump weights with the egs, and apply the
# weights to the derivative w.r.t. the output instead of using the
# --min-deriv-time and --max-deriv-time options.  Increased the frames-overlap
# to 30 also.  This wil.  give 10 frames on each side with zero derivs, then
# ramping up to a weight of 1.0 over 10 frames.

# _m is as _k but after a code change that makes the denominator FST more
# compact.  I am rerunning in order to verify that the WER is not changed (since
# it's possible in principle that due to edge effects related to weight-pushing,
# the results could be a bit different).
#  The results are inconsistently different but broadly the same.  On all of eval2000,
#  the change k->m is 20.7->20.9 with tg LM and 18.9->18.6 after rescoring.
#  On the train_dev data, the change is  19.3->18.9 with tg LM and 17.6->17.6 after rescoring.


# _k is as _i but reverting the g->h change, removing the --scale-max-param-change
# option and setting max-param-change to 1..  Using the same egs.

# _i is as _h but longer egs: 150 frames instead of 75, and
# 128 elements per minibatch instead of 256.

# _h is as _g but different application of max-param-change (use --scale-max-param-change true)

# _g is as _f but more splicing at last layer.

# _f is as _e but with 30 as the number of left phone classes instead
# of 10.

# _e is as _d but making it more similar in configuration to _b.
# (turns out b was better than a after all-- the egs' likelihoods had to
# be corrected before comparing them).
# the changes (vs. d) are: change num-pdfs target from 8k to 12k,
# multiply learning rates by 5, and set final-layer-normalize-target to 0.5.

# _d is as _c but with a modified topology (with 4 distinct states per phone
# instead of 2), and a slightly larger num-states (8000) to compensate for the
# different topology, which has more states.

# _c is as _a but getting rid of the final-layer-normalize-target (making it 1.0
# as the default) as it's not clear that it was helpful; using the old learning-rates;
# and modifying the target-num-states to 7000.

# _b is as as _a except for configuration changes: using 12k num-leaves instead of
# 5k; using 5 times larger learning rate, and --final-layer-normalize-target=0.5,
# which will make the final layer learn less fast compared with other layers.

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_2b  # Note: _sp will get added to this if $speed_perturb == true.

# TDNN options
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -6,3 -6,3"

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=30
max_param_change=1.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
frames_per_eg=150
remove_egs=false

# End configuration section.
echo "$0 $@"  # Print the command line for logging

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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5o_tree$suffix

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
  lang=data/lang_chain_d
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo2.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set data/lang_chain_d $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train_tdnn.sh --stage $train_stage \
    --apply-deriv-weights false \
    --pdf-boundary-penalty 0.0 \
    --get-egs-stage $get_egs_stage \
    --minibatch-size $minibatch_size \
    --egs-opts "--frames-overlap-per-eg 75" \
    --frames-per-eg $frames_per_eg \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --max-param-change $max_param_change \
    --final-layer-normalize-target $final_layer_normalize_target \
    --relu-dim 850 \
    --cmd "$decode_cmd" \
    --remove-egs $remove_egs \
    data/${train_set}_hires $treedir exp/tri4_lats_nodup$suffix $dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --transition-scale 0.0 \
      --self-loop-scale 0.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 14 ]; then
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;

if [ $stage -le 15 ]; then
  for decode_set in train_dev eval2000; do
      (
      iter=300
      steps/nnet3/decode.sh --iter $iter --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${decode_suff}_it$iter || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg}_it$iter || exit 1;
      fi
      ) &
  done
fi

wait;
exit 0;
