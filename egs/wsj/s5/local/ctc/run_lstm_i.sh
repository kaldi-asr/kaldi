#!/bin/bash

# i is as h, but using a monophone system and no context, to get closer to
# vanilla CTC.

# h is as g, but going for more aggressive learning,
# max-param-change=1.0, and momentum=0.75.  Also increasing the shrinking threshold so we
# will get more shrinking (we were getting some oversaturated neurons).

# g is as f, but with more epochs (5->8), changed initial-effective-lrate and
# final-effective-lrate,
# more num-jobs-final (4->6), reduced momentum from 0.9 to 0.5, and
# added the max-change to the code (default: 0.4) to hopefully stabilize it.
# removed the num-bptt-steps option which should not have been there!

# f is as e, but increased right-tolerance from 10 to 20 frames,
# chunk-width from 50 to 75, and larger final-effective-lrate.


# e is as d, but with higher learning-rates (it looks like the shrinkage was overwhelming
# the learning towards the end, in the d run).

# this is a basic ctc+lstm script
# d is as b, but:
# changing lstm-delay to "-1 -3 -3" (was -1 -1 -1 in "b", as default, but
# we since changed the script to make the default "-1 -2 -3".
# Reduced num-epochs from 10 to 5, and num-jobs-final from 12 to 4.
# Reduced parameters (cell-dim,hidden-dim: 1024->750, recurrent_projection_dim
# non_recurrent_projection_dim: 256->175.
# reduced learning rate slightly.
# added --frames-per-iter 800000 (since it was training too fast)...
#
# note: b was as a but double learning-rate.
#


set -e

# configs for ctc
treedir=exp/ctc/tri5b_tree_monophone
dir=exp/ctc/lstm_i

stage=0
train_stage=-10
splice_indexes="-2,-1,0,1,2 0 0"
num_lstm_layers=3
cell_dim=750
hidden_dim=750
recurrent_projection_dim=175
non_recurrent_projection_dim=175
chunk_width=75
chunk_left_context=30
clipping_threshold=5.0
norm_based_clipping=true
has_fisher=true

num_epochs=8
lstm_delay="-1 -3 -3"
# training options
initial_effective_lrate=0.001
final_effective_lrate=0.0002
num_jobs_initial=1
num_jobs_final=6
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=false


# End configuration section.

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

ali_dir=exp/tri4b_ali_si284

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file.
  lang=data/lang_ctc
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  utils/gen_topo.pl 1 1 $nonsilphonelist $silphonelist >$lang/topo
fi


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4b_ali_si284/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_si284 \
    data/lang exp/tri4b exp/tri4b_lats_si284
  rm exp/tri4b_lats_si284/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Starting from the alignments in tri4b_ali_si284, we train a rudimentary
  # LDA+MLLT system with a 1-state HMM topology and with only left phonetic
  # context (one phone's worth of left context, for now).  We set "--num-iters
  # 1" because we only need the tree from this system.
  steps/train_sat.sh --cmd "$train_cmd" --num-iters 1 \
    --tree-stats-opts "--collapse-pdf-classes=true" \
    --cluster-phones-opts "--pdf-class-list=0" \
    --context-opts "--context-width=1 --central-position=0" \
     2500 15000 data/train_si284 data/lang_ctc exp/tri4b_ali_si284 $treedir

  # copying the transforms is just more convenient than having the transforms in
  # a separate directory.  because we do only one iteration of estimation in
  # $treedir, it deosn't get to estimating any transforms.
  # ?? May not be needed.
  #cp exp/tri4b_ali_si284/trans.* $treedir

  # note: the likelihood improvement from building the tree is 6.49, versus 8.48
  # in the baseline.
fi

if [ $stage -le 11 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

 touch exp/ctc/lstm_f/egs/.nodelete # keep egs around when that run dies.

  # adding --target-num-history-states 500 to match the egs of run_lstm_a.sh.  The
  # script must have had a different default at that time.
  steps/nnet3/ctc/train_lstm.sh --stage $train_stage \
    --egs-dir exp/ctc/lstm_f/egs \
    --right-tolerance 20 \
    --frames-per-iter 800000 \
    --ngram-order 1 \
    --max-param-change 1.0 \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --lstm-delay "$lstm_delay" \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --shrink 0.99 \
    --shrink-threshold 0.15 \
    --momentum 0.75 \
    --cmd "$decode_cmd" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --clipping-threshold $clipping_threshold \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --norm-based-clipping $norm_based_clipping \
    --remove-egs $remove_egs \
    data/train_si284_hires data/lang_ctc $treedir exp/tri4b_lats_si284  $dir  || exit 1;
fi

phone_lm_weight=0.15
if [ $stage -le 12 ]; then
  for lm_suffix in tgpr bd_tgpr; do
    steps/nnet3/ctc/mkgraph.sh --phone-lm-weight $phone_lm_weight \
        data/lang_test_${lm_suffix} $dir $dir/graph_${lm_suffix}_${phone_lm_weight}
  done
fi


if [ $stage -le 13 ]; then
  # offline decoding
  for lm_suffix in tgpr bd_tgpr; do
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet3/ctc/decode.sh --nj 8 --cmd "$decode_cmd" \
        --frames-per-chunk $chunk_width --extra-left-context $chunk_left_context \
         $dir/graph_${lm_suffix}_${phone_lm_weight} data/test_${year}_hires \
         $dir/decode_${lm_suffix}_${year}_plm${phone_lm_weight} &
    done
  done
fi

wait

exit 0;
