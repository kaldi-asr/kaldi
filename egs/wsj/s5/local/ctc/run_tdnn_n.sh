#!/bin/bash

# n is as m but after a code fix that treats the edges better and gets more
# accurate objf values.

# m is as j but being more careful about the edges of the computation: adding
#    --left-deriv-truncate 5  --right-deriv-truncate 10 --egs-opts "--frames-overlap-per-eg 15"
#
# j is as h but setting --target-num-history-states=500.
#   I also had to reduce --num-jobs-final from 8 to 6 after encountering
#  instability :-(.
# WER is almost the same as h.

# h is as g running after changing the script to set
# --target-num-history-states=1000 by default.

#  g is as f (which failed after diverging), but fewer jobs and smaller
#  initial lrate, and one more epoch; and I also
#  changed the script to dump 800k frames (of input) instead of 400k, per
#  job, so that each job will last a bit longer.
#  not really hoping for better model, just more speed.
#  Also using the longer egs and smaller minibatch-size, as in d (run_tdnn_4.sh)
#
# f is as c (aka run_tdnn3.sh), but fewer epochs and higher learning rate

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
treedir=exp/ctc/tri5b_tree
dir=exp/ctc/nnet_tdnn_n

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

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;


if [ $stage -le 8 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file.
  lang=data/lang_ctc
  rm -r $lang 2>/dev/null
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  utils/gen_topo.pl 1 1 $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 9 ]; then
  # Starting from the alignments in tri4b_ali_si284, we train a rudimentary
  # LDA+MLLT system with a 1-state HMM topology and with only left phonetic
  # context (one phone's worth of left context, for now).  We set "--num-iters
  # 1" because we only need the tree from this system.
  steps/train_sat.sh --cmd "$train_cmd" --num-iters 1 \
    --tree-stats-opts "--collapse-pdf-classes=true" \
    --cluster-phones-opts "--pdf-class-list=0" \
    --context-opts "--context-width=2 --central-position=1" \
     2500 15000 data/train_si284 data/lang_ctc exp/tri4b_ali_si284 $treedir

  # copying the transforms is just more convenient than having the transforms in
  # a separate directory.  because we do only one iteration of estimation in
  # $treedir, it deosn't get to estimating any transforms.
  # ?? May not be needed.
  #cp exp/tri4b_ali_si284/trans.* $treedir

  # note: the likelihood improvement from building the tree is 6.49, versus 8.48
  # in the baseline.
fi

if [ $stage -le 10 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4b_ali_si284/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_si284 \
    data/lang exp/tri4b exp/tri4b_lats_si284
  rm exp/tri4b_lats_si284/fsts.*.gz # save space
fi

if [ $stage -le 11 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/ctc/train_tdnn.sh --stage $train_stage \
    --left-deriv-truncate 5  --right-deriv-truncate 10 --egs-opts "--frames-overlap-per-eg 15" \
    --target-num-history-states 500 \
    --minibatch-size 256 --frames-per-eg 50 \
    --num-epochs 5 --num-jobs-initial 2 --num-jobs-final 6 \
    --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train_si284 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.003 --final-effective-lrate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 250 \
    data/train_si284_hires data/lang_ctc $treedir exp/tri4b_lats_si284 $dir  || exit 1;
fi


phone_lm_weight=0.0
if [ $stage -le 12 ]; then
  for lm_suffix in tgpr bd_tgpr; do
    steps/nnet3/ctc/mkgraph.sh --phone-lm-weight $phone_lm_weight \
        data/lang_test_${lm_suffix} $dir $dir/graph_${lm_suffix}_${phone_lm_weight}
  done
fi

unk_penalty=0   # natural-log penalty for <UNK>, which for some reason is
                # decoded a lot in this setup.


if [ $stage -le 13 ]; then
  for lm_suffix in tgpr bd_tgpr; do
    src=$dir/graph_${lm_suffix}_${phone_lm_weight}
    dst=$dir/graph_${lm_suffix}_${phone_lm_weight}_unk${unk_penalty}
    rm -rf $dst;
    cp -r $src $dst
    unk_word=$(grep -w '<UNK>' data/lang_test_${lm_suffix}/words.txt | awk '{print $2}') || exit 1;
    fstprint $src/CTC.fst | awk -v u=$unk_word -v p=$unk_penalty '{if ($4 == u) $5 += p; print;} ' | \
      fstcompile > $dst/CTC.fst || exit 1;
  done
fi

if [ $stage -le 14 ]; then
  # offline decoding
  blank_scale=1.0
  for lm_suffix in tgpr bd_tgpr; do
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet3/ctc/decode.sh --nj 8 --cmd "$decode_cmd" --blank-scale $blank_scale \
         --online-ivector-dir exp/nnet3/ivectors_test_$year \
         $dir/graph_${lm_suffix}_${phone_lm_weight}_unk${unk_penalty} data/test_${year}_hires \
         $dir/decode_${lm_suffix}_${year}_plm${phone_lm_weight}_bs${blank_scale}_unk${unk_penalty} &
    done
  done
fi

wait;
exit 0;

# Results (almost as good as the baseline):
# grep WER exp/ctc/nnet_tdnn_g/decode_{tgpr,bd_tgpr}_{dev93,eval92}/scoring_kaldi/best_wer

# this is with phone_lm_weight 0.0, which on average seems very slightly better than 0.15.
b01:s5:  cat exp/ctc/nnet_tdnn_n/decode_{tgpr,bd_tgpr}_{eval92,dev93}*/scoring_kaldi/best_wer | grep plm0.0
%WER 6.63 [ 374 / 5643, 73 ins, 60 del, 241 sub ] exp/ctc/nnet_tdnn_n/decode_tgpr_eval92_plm0.0_bs1.0/wer_9_1.0
%WER 9.23 [ 760 / 8234, 114 ins, 152 del, 494 sub ] exp/ctc/nnet_tdnn_n/decode_tgpr_dev93_plm0.0_bs1.0/wer_9_0.5
%WER 4.22 [ 238 / 5643, 21 ins, 28 del, 189 sub ] exp/ctc/nnet_tdnn_n/decode_bd_tgpr_eval92_plm0.0_bs1.0/wer_10_1.0
%WER 7.37 [ 607 / 8234, 74 ins, 103 del, 430 sub ] exp/ctc/nnet_tdnn_n/decode_bd_tgpr_dev93_plm0.0_bs1.0/wer_10_0.0

b01:s5: steps/align_fmllr.sh: doing final alignment.
 cat exp/ctc/nnet_tdnn_n/decode_{tgpr,bd_tgpr}_{eval92,dev93}*/scoring_kaldi/best_wer | grep -v plm0.0
%WER 6.56 [ 370 / 5643, 67 ins, 70 del, 233 sub ] exp/ctc/nnet_tdnn_n/decode_tgpr_eval92_plm0.15_bs1.0/wer_9_0.5
%WER 9.35 [ 770 / 8234, 109 ins, 202 del, 459 sub ] exp/ctc/nnet_tdnn_n/decode_tgpr_dev93_plm0.15_bs1.0/wer_9_0.0
%WER 4.25 [ 240 / 5643, 20 ins, 36 del, 184 sub ] exp/ctc/nnet_tdnn_n/decode_bd_tgpr_eval92_plm0.15_bs1.0/wer_10_0.5
%WER 7.51 [ 618 / 8234, 77 ins, 113 del, 428 sub ] exp/ctc/nnet_tdnn_n/decode_bd_tgpr_dev93_plm0.15_bs1.0/wer_9_0.0

# Baseline results:
grep WER exp/nnet3/nnet_tdnn_a/decode_{tgpr,bd_tgpr}_{eval92,dev93}/scoring_kaldi/best_wer
exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/scoring_kaldi/best_wer:%WER 6.03 [ 340 / 5643, 74 ins, 20 del, 246 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/wer_13_1.0
exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/scoring_kaldi/best_wer:%WER 9.35 [ 770 / 8234, 162 ins, 84 del, 524 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/wer_11_0.5
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/scoring_kaldi/best_wer:%WER 3.81 [ 215 / 5643, 30 ins, 18 del, 167 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/wer_10_1.0
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/scoring_kaldi/best_wer:%WER 6.74 [ 555 / 8234, 69 ins, 72 del, 414 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/wer_11_0.0

