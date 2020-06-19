#!/usr/bin/env bash
#
# Copyright 2018  François Hernandez (Ubiqus)
#
# This script takes a rnnlm_dir and averages its models.
#
# Takes the default rnnlm_dir of tedlium s5_r3 recipe,
# and average the best model and the 10 previous and
# following ones (if they exist).


. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

rnnlm_dir=exp/rnnlm_lstm_tdnn_a
begin=
end=

. utils/parse_options.sh # accept options

# get the best iteration
best_iter=$(rnnlm/get_best_model.py $rnnlm_dir)

# get num_iters
info=$(grep "num_iters" $rnnlm_dir/info.txt)
num_iters=${info##*=}


# test if begin and end exist
if [ -z $begin ] && [ -z $end ]; then
    begin=$(($best_iter-10))
    end=$(($best_iter+10))
    if [ $begin -le 1 ]; then
        begin=1
    fi
    if [ ! $end -le $num_iters ]; then
        end=$num_iters
    fi
fi

# create list of models and embeddings files to merge
models=""
embeddings=""
for num in $(seq -s' ' $begin $end); do
    [ -f $rnnlm_dir/$num.raw ] && \
        models=$models" $rnnlm_dir/$num.raw"
	[ -f $rnnlm_dir/feat_embedding.$num.mat ] && \
        embeddings=$embeddings" $rnnlm_dir/feat_embedding.$num.mat"
done

# merge list of files
mkdir -p ${rnnlm_dir}_averaged
nnet3-average $models ${rnnlm_dir}_averaged/final.raw
matrix-sum --average=true $embeddings ${rnnlm_dir}_averaged/feat_embedding.final.mat

# copy other files to averaged rnnlm_dir
cp -r $rnnlm_dir/{info.txt,word_feats.txt,config,special_symbol_opts.txt} ${rnnlm_dir}_averaged

