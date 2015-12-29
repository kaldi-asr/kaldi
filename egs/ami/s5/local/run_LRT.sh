#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=joint # joint for joint training; multi for multi-output training
gmm_decode=false
dnn_stage=-100
echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_trees_L=$1
num_trees_T=$2
num_trees_R=$3
lambda=$4
num_leaves=$5
num_gauss=$6

num_trees=$[$num_trees_L+$num_trees_T+$num_trees_L]

data=data/train_si284
lang=data/lang
alidir=exp/tri4b_ali_si284
dir=exp/LRT_${num_trees_L}_${num_trees_T}_${num_trees_R}_$lambda/tri_${num_leaves}_${num_gauss}

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/train_sat_LRT.sh --cmd "$train_cmd" \
      --numtrees_L $num_trees_L \
      --numtrees_T $num_trees_T \
      --numtrees_R $num_trees_R \
      --lambda $lambda \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  utils/mkgraph.sh data/lang_test_bd_tgpr $dir/virtual $dir/virtual/graph_bd_tgpr

  if [ "$gmm_decode" == "true" ]; then
    steps/decode_multi.sh --cmd "$decode_cmd" --nj 10 \
        --numtrees $num_trees --transform_dir exp/tri4b/decode_bd_tgpr_dev93 \
        $dir/virtual/graph data/test_dev93 $dir/virtual/decode_dev93 $dir/virtual/tree-mapping &
    steps/decode_multi.sh --cmd "$decode_cmd" --nj 8 \
        --numtrees $num_trees --transform_dir exp/tri4b/decode_bd_tgpr_eval92 \
        $dir/virtual/graph data/test_eval92 $dir/virtual/decode_eval92 $dir/virtual/tree-mapping &
  fi
fi

nnet3dir=${dir}/../tdnn_${method}_${num_leaves}

#./local/nnet3/run_tdnn_$method.sh --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage

method=multi
nnet3dir=${dir}/../tdnn_${method}_${num_leaves}
#dnn_stage=81
./local/nnet3/run_tdnn_$method.sh --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
