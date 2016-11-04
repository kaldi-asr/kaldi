#!/bin/bash

# Copyright 2014  Vimal Manohar. Apache 2.0.
# This script, which will generally be called during the neural-net training
# relabels existing examples with better labels obtained by realigning the data
# with the current nnet model

# Begin configuration section
cmd=run.pl
stage=0
extra_egs=        # Names of additional egs files that need to relabelled
                  # other than egs.*.*.ark, combine.egs, train_diagnostic.egs,
                  # valid_diagnostic.egs
iter=final
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

for f in $alidir/ali.1.gz $model $egs_in_dir/egs.1.0.ark $egs_in_dir/combine.egs \
  $egs_in_dir/valid_diagnostic.egs $egs_in_dir/train_diagnostic.egs \
  $egs_in_dir/num_jobs_nnet $egs_in_dir/iters_per_epoch $egs_in_dir/samples_per_iter; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

num_jobs_nnet=`cat $egs_in_dir/num_jobs_nnet`
iters_per_epoch=`cat $egs_in_dir/iters_per_epoch`
samples_per_iter_real=`cat $egs_in_dir/samples_per_iter`
num_jobs_align=`cat $alidir/num_jobs`

mkdir -p $dir/log

echo $num_jobs_nnet > $dir/num_jobs_nnet
echo $iters_per_epoch > $dir/iters_per_epoch
echo $samples_per_iter_real > $dir/samples_per_iter

alignments=$(for n in $(seq $num_jobs_align); do echo -n "$alidir/ali.$n.gz "; done)

if [ $stage -le 0 ]; then
  egs_in=
  egs_out=
  for x in `seq 1 $num_jobs_nnet`; do
    for y in `seq 0 $[$iters_per_epoch-1]`; do
      utils/create_data_link.pl $dir/egs.$x.$y.ark
      if [ $x -eq 1 ]; then
        egs_in="$egs_in ark:$egs_in_dir/egs.JOB.$y.ark "
        egs_out="$egs_out ark:$dir/egs.JOB.$y.ark "
      fi
    done
  done

  $cmd JOB=1:$num_jobs_nnet $dir/log/relabel_egs.JOB.log \
    nnet-relabel-egs "ark:gunzip -c $alignments | ali-to-pdf $model ark:- ark:- |" \
    $egs_in $egs_out || exit 1
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
