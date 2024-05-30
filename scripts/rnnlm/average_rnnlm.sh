#!/usr/bin/env bash
#
# Created by HY Chan (2020)
# Modified from: 2018  François Hernandez (Ubiqus) tedlium/s5_r3/local/rnnlm/average_rnnlm.sh
# This script averages rnnlm models.
#
# Input the following: 1) rnnlm_dir 2) @iteration 3) number of models before @iteration 4) number of models follow @iteration
# and average the model "in previous" and "following" (if they exist).
# It usually gives small perplexity reduction by averaging "the best @iteration" model from 
# several former iterations and later iterations (if exists)


. ./cmd.sh
. ./path.sh

if [ $# -ne 4 ]; then
  echo "usage: $0 rnnlmdir iter left_dist right_dist"
  echo "usage: $0 exp/rnnlm 500 3 6"
  exit
fi

rnnlm_dir=$1
begin=
end=

user_iter=$2
leftdist=$3
rightdist=$4

. utils/parse_options.sh # accept options

# get the best iteration
find_best_iter=$(rnnlm/get_best_model.py $rnnlm_dir)
echo "automatic found best iteration @ $find_best_iter , user define iteration @ $user_iter"

# get num_iters
info=$(grep "num_iters" $rnnlm_dir/info.txt)
num_iters=${info##*=}

# test if begin and end exist
if [ -z $begin ] && [ -z $end ]; then
    begin=$(($user_iter-$leftdist))
    end=$(($user_iter+$rightdist))
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
