#!/usr/bin/env bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright (c) 2015, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
cmd=run.pl
iter=final
# End configuration section


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <degs-dir> <nnet-dir>"
  echo " e.g.: $0 exp/tri4_mpe_degs exp/tri4_mpe"
  echo ""
  echo "Performs priors adjustment either on the final iteration"
  echo "or iteration of choice of the training. The adjusted model"
  echo "filename will be suffixed by \"adj\", i.e. for the final"
  echo "iteration final.mdl will become final.adj.mdl"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --iter <iteration|final>                         # which iteration to be adjusted"
  exit 1;
fi

degs_dir=$1
dir=$2

src_model=$dir/${iter}.mdl

if [ ! -f $src_model ]; then
  echo "$0: Expecting $src_model to exist."
  exit 1
fi

if [ ! -f $degs_dir/priors_egs.1.ark ]; then
  echo "$0: Expecting $degs_dir/priors_egs.1.ark to exist."
  exit 1
fi

num_archives_priors=`cat $degs_dir/info/num_archives_priors` || {
  echo "Could not find $degs_dir/info/num_archives_priors.";
  exit 1;
}

$cmd JOB=1:$num_archives_priors $dir/log/get_post.${iter}.JOB.log \
  nnet-compute-from-egs "nnet-to-raw-nnet $src_model -|" \
  ark:$degs_dir/priors_egs.JOB.ark ark:- \| \
  matrix-sum-rows ark:- ark:- \| \
  vector-sum ark:- $dir/post.${iter}.JOB.vec || {
    echo "Error in getting posteriors for adjusting priors."
    echo "See $dir/log/get_post.${iter}.*.log";
    exit 1;
  }


$cmd $dir/log/sum_post.${iter}.log \
  vector-sum $dir/post.${iter}.*.vec $dir/post.${iter}.vec || {
    echo "Error in summing posteriors. See $dir/log/sum_post.${iter}.log";
    exit 1;
  }

rm -f $dir/post.${iter}.*.vec

echo "Re-adjusting priors based on computed posteriors for iter $iter"
$cmd $dir/log/adjust_priors.${iter}.log \
  nnet-adjust-priors $src_model $dir/post.${iter}.vec $dir/${iter}.adj.mdl || {
    echo "Error in adjusting priors. See $dir/log/adjust_priors.${iter}.log";
    exit 1;
  }

echo "Done adjusting priors (on $src_model)"
