#!/bin/bash

# This is the latest version of training that combines RM and WSJ, in a setup where
# there are no shared phones (so it's like a multilingual setup).
# Before running this script, go to ../../wsj/s5, and after running
# the earlier stages in the run.sh (so the baseline SAT system is built),
# run the following:
# 
# local/online/run_nnet2.sh --stage 8 --dir exp/nnet2_online/nnet_ms_a_partial --exit-train-stage 15    
#
# (you may want to keep --stage 8 on the above command line after run_nnet2.sh,
# in case you already ran some scripts in local/online/ in ../../wsj/s5/ and
# the earlier stages are finished, otherwise remove it).


stage=0
train_stage=-10
srcdir=../../wsj/s5/exp/nnet2_online/nnet_ms_a_partial
src_alidir=../../wsj/s5/exp/tri4b_ali_si284   # it's important that this be the alignments
                                              # we actually trained srcdir on.
src_lang=../../wsj/s5/data/lang
dir=exp/nnet2_online_wsj/nnet_ms_a
use_gpu=true
set -e

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi

# Check inputs.
for f in $srcdir/egs/egs.1.ark $srcdir/egs/info/egs_per_archive \
    ${srcdir}_online/final.mdl $src_alidir/ali.1.gz; do 
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

if ! cmp $srcdir/tree $src_alidir/tree; then
  echo "$0: trees in $srcdir and $src_alidir do not match"
  exit 1;
fi

if [ $stage -le 0 ]; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train data/train_max2
fi

if [ $stage -le 1 ]; then 
  echo "$0: dumping egs for RM data"
  steps/online/nnet2/get_egs2.sh --cmd "$train_cmd" \
    data/train_max2 exp/tri3b_ali ${srcdir}_online ${dir}/egs
fi

if [ $stage -le 2 ]; then
  echo "$0: doing the multilingual training."

  # 4 jobs for WSJ, 1 for RM; this affects the data weighting.  num-epochs is for
  # first one (WSJ).
  # the script said this:
  # steps/nnet2/train_multilang2.sh: Will train for 7 epochs (of language 0) = 140 iterations
  # steps/nnet2/train_multilang2.sh: 140 iterations is approximately 35 epochs for language 1

  # note: the arguments to the --mix-up option are (number of mixtures for WSJ,
  # number of mixtures for RM).  We just use fairly typical numbers for each
  # (although a bit fewer for WSJ, since we're not so concerned about the
  # performance of that system).

  steps/nnet2/train_multilang2.sh --num-jobs-nnet "4 1" \
    --stage $train_stage \
    --mix-up "10000 4000" \
    --cleanup false --num-epochs 7 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$train_cmd" --parallel-opts "$parallel_opts" --num-threads "$num_threads" \
    $src_alidir $srcdir/egs exp/tri3b_ali $dir/egs ${srcdir}_online/final.mdl $dir
fi


if [ $stage -le 3 ]; then
  # Prepare the RM and WSJ setups for decoding, with config files
  # (for WSJ, we need the config files for discriminative training).

  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir}_online $src_lang $dir/0 ${dir}_wsj_online

  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir}_online data/lang $dir/1 ${dir}_rm_online
fi

if [ $stage -le 4 ]; then
  # do the actual online decoding with iVectors.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_rm_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_rm_online/decode_ug || exit 1;
  wait
fi

exit 0;

