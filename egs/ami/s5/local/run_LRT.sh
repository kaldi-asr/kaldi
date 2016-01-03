#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=joint # joint for joint training; multi for multi-output training
gmm_decode=true
dnn_stage=-100
mic=ihm
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

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

data=data/$mic/train
lang=data/lang
alidir=exp/$mic/tri4a_ali
dir=exp/$mic/LRT_${num_trees_L}_${num_trees_T}_${num_trees_R}_$lambda/tri_${num_leaves}_${num_gauss}

if [ "$gmm" == "true" ]; then
  echo training GMM systems
false &&  steps/train_sat_LRT.sh --cmd "$train_cmd" \
      --numtrees_L $num_trees_L \
      --numtrees_T $num_trees_T \
      --numtrees_R $num_trees_R \
      --lambda $lambda \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

false &&  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

false &&  $highmem_cmd $dir/virtual/graph_$LM/mkgraph.log utils/mkgraph.sh ${lang}_${LM} $dir/virtual $dir/virtual/graph_$LM

nj=30

  if [ "$gmm_decode" == "true" ]; then
    steps/decode_multi.sh --cmd "$decode_cmd" --nj $nj \
        --config conf/decode.conf \
        --numtrees $num_trees --transform_dir exp/$mic/tri4a/decode_dev_$LM \
        $dir/virtual/graph_$LM data/$mic/dev $dir/virtual/decode_dev_$LM $dir/virtual/tree-mapping &
    steps/decode_multi.sh --cmd "$decode_cmd" --nj $nj \
        --config conf/decode.conf \
        --numtrees $num_trees --transform_dir exp/$mic/tri4a/decode_eval_$LM \
        $dir/virtual/graph_$LM data/$mic/eval $dir/virtual/decode_eval_$LM $dir/virtual/tree-mapping &
  fi
fi
nnet3dir=${dir}/../tdnn_${method}_${num_leaves}

./local/nnet3/run_tdnn_$method.sh --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
exit
method=multi
nnet3dir=${dir}/../tdnn_${method}_${num_leaves}
#dnn_stage=81
./local/nnet3/run_tdnn_$method.sh --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
