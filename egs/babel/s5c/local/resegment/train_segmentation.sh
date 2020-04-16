#!/usr/bin/env bash

# Copyright 2014  Vimal Manohar, Johns Hopkins University (Author: Jan Trmal)
# Apache 2.0

nj=16             # nj for training subset of whole
cmd=run.pl        # How to run the parallel tasks
boost_sil=1.0
ext_alidir=       # Use this alignment directory instead for getting new one

# End of configuration

. utils/parse_options.sh

set -o pipefail
set -e
set -u
if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <in-model-dir> <data-dir> <lang-dir> <out-model-dir>"
  echo " e.g.:"
  echo "$0 exp/tri4 data/train data/lang exp/tri4b_seg"
  echo " Options (selection. For full options, see the script itself):"
  echo "    --nj <numjobs|16>            # Number of parallel jobs"
  echo "    --cmd <cmdstring|run.pl>     # How to run the parallel tasks"
  exit 1
fi

in_model_dir=$1          # Model used for alignment
train_data_dir=$2
lang=$3
out_model_dir=$4

if [ ! -d $train_data_dir ] ; then
  echo "$0: Unable to find directory $train_data_dir."
  echo "$0: Run run-0-fillers.sh or run-1-main.sh first to prepare data directory"
  exit 1
fi

# Align train_whole_sub3 using tri4 models and train a LDA + MLLT model
# on it.
alidir=${in_model_dir}_train_seg_ali

if [ ! -z $ext_alidir ] && [ -s $ext_alidir/ali.1.gz ]; then
  alidir=$ext_alidir
elif [ ! -f $alidir/.done ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" --boost-silence $boost_sil \
    $train_data_dir $lang $in_model_dir $alidir || exit 1;
  touch $alidir/.done
fi

if [ ! -f $out_model_dir/.done ]; then
  steps/train_lda_mllt.sh --cmd "$cmd" --realign-iters "" --boost-silence $boost_sil \
    1000 10000 $train_data_dir $lang $alidir $out_model_dir || exit 1;
  touch $out_model_dir/.done
fi

if [ ! -f $out_model_dir/graph.done ]; then
  # Make the phone decoding-graph.
  steps/make_phone_graph.sh $lang $alidir $out_model_dir || exit 1;
  utils/mkgraph.sh $lang $out_model_dir $out_model_dir/graph | \
    tee $out_model_dir/mkgraph.log || exit 1
  touch $out_model_dir/graph.done
fi
