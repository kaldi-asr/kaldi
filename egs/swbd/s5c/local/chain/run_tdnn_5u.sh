#!/bin/bash

# _5u is as _5o but modifying the mfcc generation to use a narrower window while
# generating the lower-order mfcc coefficients (the first 10).

# Abandoning it partway through after I got the following less-than-promising diagnostics.
# grep Overall exp/chain/tdnn_5{o,u}_sp/log/compute_prob_valid.84.log | grep -v xent
# exp/chain/tdnn_5o_sp/log/compute_prob_valid.84.log:LOG (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:175) Overall log-probability for 'output' is -0.146977 + -0.0159528 = -0.16293 per frame, over 20000 frames.
# exp/chain/tdnn_5u_sp/log/compute_prob_valid.84.log:LOG (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:175) Overall log-probability for 'output' is -0.147207 + -0.015692 = -0.162899 per frame, over 20000 frames.
# a03:s5c: grep Overall exp/chain/tdnn_5{o,u}_sp/log/compute_prob_train.84.log | grep -v xent
# exp/chain/tdnn_5o_sp/log/compute_prob_train.84.log:LOG (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:175) Overall log-probability for 'output' is -0.146703 + -0.0165036 = -0.163207 per frame, over 20000 frames.
# exp/chain/tdnn_5u_sp/log/compute_prob_train.84.log:LOG (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:175) Overall log-probability for 'output' is -0.145524 + -0.0162272 = -0.161751 per frame, over 20000 frames.

# _5o is as _5n but adding an extra splicing layer and increasing the
# splice-width slightly on the 1st layer, to get closer to the context in 5n;
# having one more layer running at double-frequency, and reverting the frame-length to
# the same as in the baseline (25ms) to avoid sacrificing frequency resolution.

# Objective functions improve but WER change is quite small vs 5n (~0.1%).  so
# not clear that the extra time is worth it (it's noticeably slower to train as
# that extra layer is at a higher sampling rate).
#
#System                       5j        5n        5o
#WER on train_dev(tg)      17.59     16.85     16.83
#WER on train_dev(fg)      16.33     15.67     15.60
#WER on eval2000(tg)        19.1      19.1      18.8
#WER on eval2000(fg)        17.5      17.3      17.2
#Final train prob      -0.114691 -0.116341 -0.111613
#Final valid prob      -0.130761 -0.130884 -0.126765

# _5n is as _5j (also omitting the iVectors), but using double the input frame
# rate from 10 to 5 ms (and reducing frame width from 25 to 20), and modifying
# the splice indexes accordingly

# _5j is as _5e, but omitting the iVectors.

# Definitely worse, although curiously, there is very little effect on the valid prob.
#./compare_wer.sh 5e 5j
#System                       5e        5j
#WER on train_dev(tg)      15.43     17.59
#WER on train_dev(fg)      14.32     16.33
#WER on eval2000(tg)        17.3      19.1
#WER on eval2000(fg)        15.5      17.5
#Final train prob      -0.110056 -0.114691
#Final valid prob      -0.129184 -0.130761


# _5e is as _5b, but reducing --xent-regularize from 0.2 to 0.1 (since based on
# the results of 4v, 4w and 5c, it looks like 0.1 is better than 0.2 or 0.05).

# The improvement is small but consistent (0.1, 0.1, 0.0, 0.1) and also seen
# in the train and valid probs.
#System                       5b        5e
#WER on train_dev(tg)      15.51     15.43
#WER on train_dev(fg)      14.39     14.32
#WER on eval2000(tg)        17.3      17.3
#WER on eval2000(fg)        15.6      15.5
#Final train prob      -0.112013 -0.110056
#Final valid prob      -0.130879 -0.129184

# _5b is as _5a, but adding --leaky-hmm-coefficient 0.1.

# It does seem helpful on average: (-0.35, -0.35, -0.1, 0).
#./compare_wer.sh 5a 5b
#System                       5a        5b
#WER on train_dev(tg)      15.86     15.51
#WER on train_dev(fg)      14.74     14.39
#WER on eval2000(tg)        17.4      17.3
#WER on eval2000(fg)        15.6      15.6
#Final train prob     -0.0998359 -0.112013
#Final valid prob      -0.115884 -0.130879

# _5a is as _4w, but increasing jesus-forward-output-dim from 1400 to 1800, and
# jesus-forward-input-dim from 400 to 500.  Hoping that the cross-entropy regularization
# will mean that the increased parameters are now helpful.

# _4w is as _4v, but doubling --xent-regularize to 0.2

# _4v is as _4r, but with --xent-regularize 0.1.  Increasing max_param_change
# from 1.0 to 2.0 because there is a lot of parameter change in the final xent
# layer, and this limits the rate of change of the other layers.

# _4r is as _4f, but one more hidden layer, and reducing context of existing
# layers so we can re-use the egs.  Reducing jesus-forward-output-dim slightly
# from 1500 to 1400.

# This is better than 4f by almost all metrics.
# ./compare_wer.sh 4f 4r
# System                       4f        4r
# WER on train_dev(tg)      16.83     16.50
# WER on train_dev(fg)      15.73     15.45
# WER on eval2000(tg)        18.4      18.3
# WER on eval2000(fg)        16.6      16.7
# Final train prob      -0.105832 -0.103652
# Final valid prob      -0.123021 -0.121105

# _4f is as _4e, but halving the regularization from 0.0001 to 0.00005.

# It's even better than 4e, by about 0.3% abs.
#                        4c    4e      4f
#  Final valid prob:   -0.1241 -0.1267  -0.1230
#  Final train prob:   -0.08820 -0.1149 -0.1058

# ./show_wer.sh 4f
# %WER 16.83 [ 8282 / 49204, 870 ins, 2354 del, 5058 sub ] exp/chain/tdnn_4f_sp/decode_train_dev_sw1_tg/wer_10_0.0
# %WER 15.73 [ 7739 / 49204, 864 ins, 2256 del, 4619 sub ] exp/chain/tdnn_4f_sp/decode_train_dev_sw1_fsh_fg/wer_10_0.0
# %WER 18.4 | 4459 42989 | 83.5 11.0 5.5 2.0 18.4 56.2 | exp/chain/tdnn_4f_sp/decode_eval2000_sw1_tg/score_9_0.0/eval2000_hires.ctm.filt.sys
# %WER 16.6 | 4459 42989 | 85.2 9.7 5.1 1.8 16.6 53.4 | exp/chain/tdnn_4f_sp/decode_eval2000_sw1_fsh_fg/score_9_0.0/eval2000_hires.ctm.filt.sys
# a03:s5c: ./show_wer.sh 4e
# %WER 17.09 [ 8407 / 49204, 923 ins, 2242 del, 5242 sub ] exp/chain/tdnn_4e_sp/decode_train_dev_sw1_tg/wer_9_0.0
# %WER 15.91 [ 7829 / 49204, 932 ins, 2141 del, 4756 sub ] exp/chain/tdnn_4e_sp/decode_train_dev_sw1_fsh_fg/wer_9_0.0
# %WER 18.5 | 4459 42989 | 83.5 10.8 5.7 2.0 18.5 56.0 | exp/chain/tdnn_4e_sp/decode_eval2000_sw1_tg/score_9_0.0/eval2000_hires.ctm.filt.sys
# %WER 16.9 | 4459 42989 | 84.9 9.8 5.4 1.8 16.9 53.9 | exp/chain/tdnn_4e_sp/decode_eval2000_sw1_fsh_fg/score_9_0.0/eval2000_hires.ctm.filt.sys


# _4e is as _4c, but adding the option --l2-regularize 0.0001.

# _4c is as _4a, but using half the --jesus-hidden-dim: 7500 versus 15000.

# _4a is as _3s, but using narrower splice-indexes in the first layer.

# _3s is as _3r but reducing jesus-forward-input-dim from 500 to 400.
# num-params is quite small now: 5.4 million, vs. 12.1 million in 2y, and 8.8 million in 3p.
# This of course reduces overtraining.  Results are a bit better than 3p but still
# not as good as 2y

# ./show_wer.sh 3s
# %WER 17.88 [ 8799 / 49204, 1006 ins, 2312 del, 5481 sub ] exp/chain/tdnn_3s_sp/decode_train_dev_sw1_tg/wer_11_0.0
# %WER 16.67 [ 8200 / 49204, 982 ins, 2221 del, 4997 sub ] exp/chain/tdnn_3s_sp/decode_train_dev_sw1_fsh_fg/wer_11_0.0
# %WER 19.6 | 4459 42989 | 82.8 11.8 5.4 2.4 19.6 57.6 | exp/chain/tdnn_3s_sp/decode_eval2000_sw1_tg/score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 17.6 | 4459 42989 | 84.4 10.1 5.4 2.1 17.6 54.7 | exp/chain/tdnn_3s_sp/decode_eval2000_sw1_fsh_fg/score_11_0.0/eval2000_hires.ctm.filt.sys
# a03:s5c: ./show_wer.sh 3p
# %WER 18.05 [ 8880 / 49204, 966 ins, 2447 del, 5467 sub ] exp/chain/tdnn_3p_sp/decode_train_dev_sw1_tg/wer_12_0.0
# %WER 16.86 [ 8296 / 49204, 967 ins, 2321 del, 5008 sub ] exp/chain/tdnn_3p_sp/decode_train_dev_sw1_fsh_fg/wer_12_0.0
# %WER 19.8 | 4459 42989 | 82.4 11.5 6.1 2.1 19.8 57.7 | exp/chain/tdnn_3p_sp/decode_eval2000_sw1_tg/score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 18.2 | 4459 42989 | 83.9 10.5 5.7 2.0 18.2 55.6 | exp/chain/tdnn_3p_sp/decode_eval2000_sw1_fsh_fg/score_11_0.0/eval2000_hires.ctm.filt.sys
# a03:s5c: ./show_wer.sh 2y
# %WER 16.99 [ 8358 / 49204, 973 ins, 2193 del, 5192 sub ] exp/chain/tdnn_2y_sp/decode_train_dev_sw1_tg/wer_11_0.0
# %WER 15.86 [ 7803 / 49204, 959 ins, 2105 del, 4739 sub ] exp/chain/tdnn_2y_sp/decode_train_dev_sw1_fsh_fg/wer_11_0.0
# %WER 18.9 | 4459 42989 | 83.4 11.3 5.3 2.3 18.9 56.3 | exp/chain/tdnn_2y_sp/decode_eval2000_sw1_tg/score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 17.0 | 4459 42989 | 85.1 10.1 4.8 2.1 17.0 53.5 | exp/chain/tdnn_2y_sp/decode_eval2000_sw1_fsh_fg/score_10_0.0/eval2000_hires.ctm.filt.sys


# _3r is as _3p but reducing the number of parameters as it seemed to be
# overtraining (despite already being quite a small model): [600,1800 ->
# 500,1500].  Also in the interim there was a script change to
# nnet3/chain/train_tdnn.sh to, on mix-up iters, apply half the max-change.
# [changing it right now from 1/2 to 1/sqrt(2) which is more consistent
# with the halving of the minibatch size.]


# _3p is the same as 3o, but after a code and script change so we can use
# natural gradient for the RepeatedAffineComponent.
# [natural gradient was helpful, based on logs;
# also made a change to use positive bias for the jesus-component affine parts.]

# _3o is as _3n but filling in the first splice-indexes from -1,2 to -1,0,1,2.

# _3n is as _3d (a non-recurrent setup), but using the more recent scripts that support
# recurrence, with improvements to the learning of the jesus layers.

# _3g is as _3f but using 100 blocks instead of 200, as in d->e 200 groups was found
# to be worse.
# It's maybe a little better than the baseline 2y; and better than 3d [-> I guess recurrence
# is helpful.]
#./show_wer.sh 3g
#%WER 17.05 [ 8387 / 49204, 905 ins, 2386 del, 5096 sub ] exp/chain/tdnn_3g_sp/decode_train_dev_sw1_tg/wer_11_0.0
#%WER 15.67 [ 7712 / 49204, 882 ins, 2250 del, 4580 sub ] exp/chain/tdnn_3g_sp/decode_train_dev_sw1_fsh_fg/wer_11_0.0
#%WER 18.7 | 4459 42989 | 83.5 11.1 5.3 2.2 18.7 56.2 | exp/chain/tdnn_3g_sp/decode_eval2000_sw1_tg/score_10_0.0/eval2000_hires.ctm.filt.sys
#%WER 16.8 | 4459 42989 | 85.1 9.9 5.0 2.0 16.8 53.7 | exp/chain/tdnn_3g_sp/decode_eval2000_sw1_fsh_fg/score_10_0.5/eval2000_hires.ctm.filt.sys
#a03:s5c: ./show_wer.sh 2y
#%WER 16.99 [ 8358 / 49204, 973 ins, 2193 del, 5192 sub ] exp/chain/tdnn_2y_sp/decode_train_dev_sw1_tg/wer_11_0.0
#%WER 15.86 [ 7803 / 49204, 959 ins, 2105 del, 4739 sub ] exp/chain/tdnn_2y_sp/decode_train_dev_sw1_fsh_fg/wer_11_0.0
#%WER 18.9 | 4459 42989 | 83.4 11.3 5.3 2.3 18.9 56.3 | exp/chain/tdnn_2y_sp/decode_eval2000_sw1_tg/score_10_0.0/eval2000_hires.ctm.filt.sys
#%WER 17.0 | 4459 42989 | 85.1 10.1 4.8 2.1 17.0 53.5 | exp/chain/tdnn_2y_sp/decode_eval2000_sw1_fsh_fg/score_10_0.0/eval2000_hires.ctm.filt.sys

#a03:s5c: ./show_wer.sh 3d
#%WER 17.35 [ 8539 / 49204, 1023 ins, 2155 del, 5361 sub ] exp/chain/tdnn_3d_sp/decode_train_dev_sw1_tg/wer_10_0.0
#%WER 16.09 [ 7919 / 49204, 1012 ins, 2071 del, 4836 sub ] exp/chain/tdnn_3d_sp/decode_train_dev_sw1_fsh_fg/wer_10_0.0
#%WER 18.9 | 4459 42989 | 83.2 11.2 5.6 2.1 18.9 56.6 | exp/chain/tdnn_3d_sp/decode_eval2000_sw1_tg/score_10_0.0/eval2000_hires.ctm.filt.sys
#%WER 17.0 | 4459 42989 | 85.0 9.8 5.2 2.0 17.0 53.6 | exp/chain/tdnn_3d_sp/decode_eval2000_sw1_fsh_fg/score_10_0.0/eval2000_hires.ctm.filt.sys

# _3f is as _3e, but modifying the splicing setup to add (left) recurrence:
# added the :3's in   --splice-indexes "-2,-1,0,1,2 -1,2 -3,0,3:-3 -6,-3,0,3:-3 -6,-3,0,3:-3"
# Therefore it's
# no longer really a tdnn, more like an RNN combined with TDNN.  BTW, I'm not re-dumping egs with extra
# context, and this isn't really ideal - I want to see if this seems promising first.

# _3e is as _3d, but increasing the --num-jesus-blocks from 100 (the default)
# to 200 in order to reduce computation in the Jesus layer.

# _3d is as _2y, and re-using the egs, but using --jesus-opts and
# configs from make_jesus_configs.py.
#  --jesus-opts "--affine-output-dim 600 --jesus-output-dim 1800 --jesus-hidden-dim 15000" \
#   --splice-indexes "-2,-1,0,1,2 -1,2 -3,0,3 -6,-3,0,3 -6,-3,0,3"

# _2y is as _2o, but increasing the --frames-per-iter by a factor of 1.5, from
# 800k to 1.2 million.  The aim is to avoid some of the per-job overhead
# (model-averaging, etc.), since each iteration takes only a minute or so.
#  I added the results to the table below.  It seems the same on average-
# which is good.  We'll probably keep this configuration.

# _2o is as _2m, but going back to our original 2-state topology, which it turns
# out that I never tested to WER.
# hm--- it's about the same, or maybe slightly better!
# caution: accidentally overwrote most of this dir, but kept the key stuff.

# note: when I compare with the rerun of 2o (not shown), this run is actually
# better.
# WER on          2m        2o          2y    [ now comparing 2o->2y:]
# train_dev,tg    17.22     17.24       16.99  0.2% better
# train_dev,fg    15.87     15.93       15.86  0.1% better
# eval2000,tg     18.7      18.7        18.9   0.2% worse
# eval2000,fg     17.0      16.9        17.0   0.1% worse

# train-prob,final  -0.0803   -0.0835
# valid-prob,final  -0.0116   -0.0122

# _2m is as _2k, but setting --leftmost-questions-truncate=-1, i.e. disabling
# that mechanism.

# _2k is as _2i, but doing the same change as in _s -> _2e, in which we
#  set --apply-deriv-weights false and --frames-overlap-per-eg 0.

# _2i is as _2d but with a new set of code for estimating the LM, in which we compute
# the log-like change when deciding which states to back off.  The code is not the same
# as the one in 2{f,g,h}.  We have only the options --num-extra-lm-states=2000.  By
# default it estimates a 4-gram, with 3-gram as the no-prune order.  So the configuration
# is quite similar to 2d, except new/more-exact code is used.

# _2d is as _2c but with different LM options:
# --lm-opts "--ngram-order=4 --leftmost-context-questions=/dev/null --num-extra-states=2000"
# ... this gives us a kind of pruned 4-gram language model, instead of a 3-gram.
# the --leftmost-context-questions=/dev/null option overrides the leftmost-context-questions
# provided from the tree-building, and effectively puts the leftmost context position as a single
# set.
#   This seems definitely helpful: on train_dev, with tg improvement is 18.12->17.55 and with fg
#  from 16.73->16.14; and on eval2000, with tg from 19.8->19.5 and with fg from 17.8->17.6.

# _2c is as _2a but after a code change in which we start using transition-scale
# and self-loop-scale of 1 instead of zero in training; we change the options to
# mkgraph used in testing, to set the scale to 1.0.  This shouldn't affect
# results at all; it's is mainly for convenience in pushing weights in graphs,
# and checking that graphs are stochastic.

# _2a is as _z but setting --lm-opts "--num-extra-states=8000".

# _z is as _x but setting  --lm-opts "--num-extra-states=2000".
# (see also y, which has --num-extra-states=500).

# _x is as _s but setting  --lm-opts "--num-extra-states=0".
#  this is a kind of repeat of the u->v experiment, where it seemed to make things
#  worse, but there were other factors involved in that so I want to be sure.

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
stage=13
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_5u # Note: _sp will get added to this if $speed_perturb == true.

# training options
num_epochs=2  # this is about the same amount of compute as the normal 4, since one
              # epoch encompasses all frame-shifts of the data.
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
frames_per_eg=300 # doubling it, since we have half the frame rate.
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

# Generate double-frame-rate version of the data with normal window size.
if [ $stage -le 12 ]; then
  mfccdir=mfcc
  for dataset in eval2000 train_dev ${train_set}; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires_dbl2
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 --mfcc-config conf/mfcc_hires_dbl2.conf \
        data/${dataset}_hires_dbl2 exp/make_hires_dbl2/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires_dbl2 exp/make_hires_dbl2/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires_dbl2  # remove segments with problems
  done
fi

# Generate double-frame-rate version of the data with smaller than normal window size;
# and only keeping the first 10 MFCC coefficients.
if [ $stage -le 13 ]; then
  mfccdir=mfcc
  for dataset in eval2000 train_dev ${train_set}; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_dbl3
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 --mfcc-config conf/mfcc_dbl3.conf \
        data/${dataset}_dbl3 exp/make_dbl3/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_dbl3  # remove segments with problems
  done
fi

# select dimension 10-39 of the dbl2 features, then create pasted features consisting
# of the 10 dimensions of the dbl3, plus the selected dimensions 10-39 of dbl2.
if [ $stage -le 14 ]; then
  mfccdir=mfcc
  for dataset in eval2000 train_dev ${train_set}; do
    steps/select_feats.sh --cmd "$train_cmd --max-jobs-run 4" 10-39 data/${dataset}_hires_dbl2 data/${dataset}_hires_dbl2_select \
          exp/make_dbl3/$dataset $mfccdir
    rm data/${dataset}_hires_dbl2_select/cmvn.scp 2>/dev/null || true
    steps/paste_feats.sh --cmd "$train_cmd --max-jobs-run 4" data/${dataset}_hires_dbl2_select data/${dataset}_dbl3 data/${dataset}_pasted \
          exp/make_dbl3/$dataset $mfccdir
    steps/compute_cmvn_stats.sh data/${dataset}_pasted exp/make_dbl3/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_pasted
  done
fi


if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train_tdnn.sh --stage $train_stage \
    --frame-subsampling-factor 6 \
    --alignment-subsampling-factor 3 \
    --xent-regularize 0.1 \
    --leaky-hmm-coefficient 0.1 \
    --l2-regularize 0.00005 \
    --jesus-opts "--jesus-forward-input-dim 500  --jesus-forward-output-dim 1800 --jesus-hidden-dim 7500 --jesus-stddev-scale 0.2 --final-layer-learning-rate-factor 0.25" \
    --splice-indexes "-1,0,1 -2,-1,0,1,2 -2,0,2 -4,-2,0,2 -6,0,6 -6,0,6 -12,-6,0" \
    --apply-deriv-weights false \
    --frames-per-iter 2400000 \
    --lm-opts "--num-extra-lm-states=2000" \
    --get-egs-stage $get_egs_stage \
    --minibatch-size $minibatch_size \
    --egs-opts "--frames-overlap-per-eg 0" \
    --frames-per-eg $frames_per_eg \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --max-param-change $max_param_change \
    --cmd "$decode_cmd" \
    --remove-egs $remove_egs \
    data/${train_set}_pasted $treedir exp/tri4_lats_nodup$suffix $dir  || exit 1;

 echo "0.005" > $dir/frame_shift # this lets the sclite decoding script know
                                 # what the frame shift was, in seconds.
fi

if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 17 ]; then
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
         --extra-left-context 20 \
          --nj 50 --cmd "$decode_cmd" \
         $graph_dir data/${decode_set}_pasted $dir/decode_${decode_set}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_pasted \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
