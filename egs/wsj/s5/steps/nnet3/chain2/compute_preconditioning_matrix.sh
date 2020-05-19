#!/bin/bash

# Copyright 2019 Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.

rand_prune=4.0
nj=8
cmd=run.pl
lda_acc_opts=
lda_transform_opts=
lda_sum_opts=
egs_opts=
stage=0
use_scp=true

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 [opts] <model> <egs-folder> <lda-output-folder>" 
    echo "e.g. $0 exp/chain/tdnn1a_sp/configs/init.raw exp/chain/tdnn1a_sp/egs/ exp/chain/tdnn1a_sp"
    echo ""
    echo "This script computes pre-conditioning matrix given the model (usually init.raw file from the config folder),"
    echo "egs-folder which has train.*.scp files to be used to train LDA, and"
    echo "lda-output-folder that will contain lda.mat file."
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
    echo "  --nj <int;8> # number of jobs. this is also the number of train.*.scp files in egs/"
    echo "  --lda-acc-opts # options to be passed to nnet3-chain-acc-lda-stats"
    echo "  --lda-sum-opts # options to be passed to sum-lda-accs"
    echo "  --lda-transform-opts # options to be passed to nnet-get-feature-transform"
    exit 1;
fi

model=$1
egs=$2
ldafolder=$3

if [ ! -d $ldafolder ]; then
    echo "Creating $ldafolder"
    mkdir -p $ldafolder || exit 1
fi


if [ $stage -le 0 ]; then
        if $use_scp; then
            egs_rspecifier="ark:nnet3-chain-copy-egs $egs_opts scp:$egs/train.JOB.scp ark:- |"
        else
            egs_rspecifier="ark:nnet3-chain-copy-egs $egs_opts ark:$egs/train.JOB.ark ark:- |"
        fi
        echo "$0: Accumulating LDA stats"
        $cmd JOB=1:$nj $ldafolder/log/acc.JOB.log \
                nnet3-chain-acc-lda-stats $lda_acc_opts --rand-prune=${rand_prune} \
                $model "${egs_rspecifier}" \
                $ldafolder/JOB.lda_stats || exit 1
fi

if [ $stage -le 1 ]; then
    echo "$0: Summing LDA stats"
    lda_stats_files=
    for i in `seq 1 $nj`; do
        lda_stats_files="$lda_stats_files $ldafolder/$i.lda_stats"
    done

    $cmd $ldafolder/log/sum_transform_stats.log \
        sum-lda-accs $lda_sum_opts $ldafolder/lda_stats $lda_stats_files || exit 1
    rm $lda_stats_files
fi

if [ $stage -le 2 ]; then
    echo "$0: Computing LDA transform"
    $cmd $ldafolder/log/get_transform.log \
        nnet-get-feature-transform $lda_transform_opts \
        $ldafolder/lda.mat $ldafolder/lda_stats || exit 1

    rm $ldafolder/lda_stats
    ln -rs $ldafolder/lda.mat $ldafolder/configs/lda.mat
fi

echo "$0: Finished computing LDA transform"
exit 0;
