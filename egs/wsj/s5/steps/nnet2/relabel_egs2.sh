#!/usr/bin/env bash

# Copyright 2014  Vimal Manohar.
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.
#
# This script, which will generally be called during the neural-net training 
# relabels existing examples with better labels obtained by realigning the data
# with the current nnet model.
# This script is as relabel_egs.sh, but is adapted to work with the newer
# egs format that is written by get_egs2.sh

# Begin configuration section
cmd=run.pl
stage=0
extra_egs=        # Names of additional egs files that need to relabelled 
                  # other than egs.*.*.ark, combine.egs, train_diagnostic.egs,
                  # valid_diagnostic.egs
iter=final
parallel_opts=
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/nnet2/relabel_egs.sh [opts] <ali-dir> <egs-in-dir> <egs-out-dir>"
  echo "  e.g: steps/nnet2/relabel_egs.sh exp/tri6_nnet/ali_1.5 exp/tri6_nnet/egs exp/tri6_nnet/egs_1.5"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."

  exit 1;
fi

alidir=$1
egs_in_dir=$2
dir=$3

model=$alidir/$iter.mdl

# Check some files.

[ -f $egs_in_dir/iters_per_epoch ] && \
  echo "$0: this script does not work with the old egs directory format" && exit 1;

for f in $alidir/ali.1.gz $model $egs_in_dir/egs.1.ark $egs_in_dir/combine.egs \
  $egs_in_dir/valid_diagnostic.egs $egs_in_dir/train_diagnostic.egs \
  $egs_in_dir/info/num_archives; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

num_archives=$(cat $egs_in_dir/info/num_archives) || exit 1;
num_jobs_align=$(cat $alidir/num_jobs) || exit 1;

mkdir -p $dir/log

mkdir -p $dir/info
cp -r $egs_in_dir/info/*  $dir/info

alignments=$(for n in $(seq $num_jobs_align); do echo $alidir/ali.$n.gz; done)

if [ $stage -le 0 ]; then
  for x in $(seq $num_archives); do
    # if $dir/storage exists, make the soft links that we'll
    # use to distribute the data across machines
    utils/create_data_link.pl $dir/egs.$x.ark
  done

  $cmd $parallel_opts JOB=1:$num_archives $dir/log/relabel_egs.JOB.log \
    nnet-relabel-egs "ark:gunzip -c $alignments | ali-to-pdf $model ark:- ark:- |" \
     ark:$egs_in_dir/egs.JOB.ark ark:$dir/egs.JOB.ark || exit 1
fi

if [ $stage -le 1 ]; then
  egs_in=
  egs_out=
  for x in combine.egs valid_diagnostic.egs train_diagnostic.egs $extra_egs; do
    utils/create_data_link.pl $dir/$x
    egs_in="$egs_in ark:$egs_in_dir/$x"
    egs_out="$egs_out ark:$dir/$x"
  done

  $cmd $dir/log/relabel_egs_extra.log \
    nnet-relabel-egs "ark:gunzip -c $alignments | ali-to-pdf $model ark:- ark:- |" \
    $egs_in $egs_out || exit 1
fi

echo "$0: Finished relabeling training examples"
