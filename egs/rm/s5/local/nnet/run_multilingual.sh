#!/bin/bash

# Copyright 2015  University of Illinois (Author: Amit Das)
# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)

# Apache 2.0

# This example script trains Multi-lingual DNN with <BlockSoftmax> output, using FBANK features.
# The network is trained on multiple languages simultaneously, creating a separate softmax layer
# per language while sharing hidden layers across all languages.
# The script supports arbitrary number of languages.

. ./cmd.sh
. ./path.sh

# Example setup, the options are in 'csl' format, they must have same number of elements,
lang_code_csl="rm,wsj" # One label for each language,
lang_weight_csl="1.0,0.1" # Per-language weights, they scale loss-function and gradient, 1.0 for each language is good,
ali_dir_csl="exp/tri3b_ali,../../wsj/s5/exp/tri4b_ali_si284" # One ali-dir per language,
data_dir_csl="data/train,../../wsj/s5/data/train_si284" # One train-data-dir per language (features will be re-computed),

nnet_type=dnn_small # dnn_small | dnn | bn

stage=0
. utils/parse_options.sh || exit 1;

set -euxo pipefail

# Convert 'csl' to bash array (accept separators ',' ':'),
lang_code=($(echo $lang_code_csl | tr ',:' ' ')) 
ali_dir=($(echo $ali_dir_csl | tr ',:' ' '))
data_dir=($(echo $data_dir_csl | tr ',:' ' '))

# Make sure we have same number of items in lists,
! [ ${#lang_code[@]} -eq ${#ali_dir[@]} -a ${#lang_code[@]} -eq ${#data_dir[@]} ] && \
  echo "Non-matching number of 'csl' items: lang_code ${#lang_code[@]}, ali_dir ${ali_dir[@]}, data_dir ${#data_dir[@]}" && \
  exit 1
num_langs=${#lang_code[@]}

# Check if all the input directories exist,
for i in $(seq 0 $[num_langs-1]); do
  echo "lang = ${lang_code[$i]}, alidir = ${ali_dir[$i]}, datadir = ${data_dir[$i]}"
  [ ! -d ${ali_dir[$i]} ] && echo  "Missing ${ali_dir[$i]}" && exit 1
  [ ! -d ${data_dir[$i]} ] && echo "Missing ${data_dir[$i]}" && exit 1
done

# Make the features,
data=data-fbank-multilingual${num_langs}-$(echo $lang_code_csl | tr ',' '-')
data_tr90=$data/combined_tr90
data_cv10=$data/combined_cv10
if [ $stage -le 0 ]; then
  # Make local copy of data-dirs (while adding language-code),
  tr90=""
  cv10=""
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[$i]}
    dir=${data_dir[$i]}
    tgt_dir=$data/${code}_$(basename $dir)
    utils/copy_data_dir.sh --utt-suffix _$code --spk-suffix _$code $dir $tgt_dir; rm $tgt_dir/{feats,cmvn}.scp || true # remove features,
    # extract features, get cmvn stats,
    steps/make_fbank_pitch.sh --nj 30 --cmd "$train_cmd --max-jobs-run 10" $tgt_dir{,/log,/data}
    steps/compute_cmvn_stats.sh $tgt_dir{,/log,/data}
    # split lists 90% train / 10% held-out,
    utils/subset_data_dir_tr_cv.sh $tgt_dir ${tgt_dir}_tr90 ${tgt_dir}_cv10
    tr90="$tr90 ${tgt_dir}_tr90"
    cv10="$cv10 ${tgt_dir}_cv10"
  done
  # Merge the datasets,
  utils/combine_data.sh $data_tr90 $tr90
  utils/combine_data.sh $data_cv10 $cv10
  # Validate,
  utils/validate_data_dir.sh $data_tr90  
  utils/validate_data_dir.sh $data_cv10  
fi

# Extract the tied-state numbers from transition models,
for i in $(seq 0 $[num_langs-1]); do
  ali_dim[i]=$(hmm-info ${ali_dir[i]}/final.mdl | grep pdfs | awk '{ print $NF }')
done
ali_dim_csl=$(echo ${ali_dim[@]} | tr ' ' ',')

# Total number of DNN outputs (sum of all per-language blocks),
output_dim=$(echo ${ali_dim[@]} | tr ' ' '\n' | awk '{ sum += $i; } END{ print sum; }')
echo "Total number of DNN outputs: $output_dim = $(echo ${ali_dim[@]} | sed 's: : + :g')"

# Objective function string (per-language weights are imported from '$lang_weight_csl'),
objective_function="multitask$(echo ${ali_dim[@]} | tr ' ' '\n' | \
  awk -v w=$lang_weight_csl 'BEGIN{ split(w,w_arr,/[,:]/); } { printf(",xent,%d,%s", $1, w_arr[NR]); }')"
echo "Multitask objective function: $objective_function"

# DNN training will be in $dir, the alignments are prepared beforehand,
dir=exp/dnn4g-multilingual${num_langs}-$(echo $lang_code_csl | tr ',' '-')-${nnet_type} 
[ ! -e $dir ] && mkdir -p $dir
echo "$lang_code_csl" >$dir/lang_code_csl
echo "$ali_dir_csl" >$dir/ali_dir_csl
echo "$data_dir_csl" >$dir/data_dir_csl
echo "$ali_dim_csl" >$dir/ali_dim_csl
echo "$objective_function" >$dir/objective_function

# Prepare the merged targets,
if [ $stage -le 1 ]; then
  [ ! -e $dir/ali-post ] && mkdir -p $dir/ali-post
  # re-saving the ali in posterior format, indexed by 'scp',
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[$i]}
    ali=${ali_dir[$i]}
    # utt suffix added by 'awk',
    ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | awk -v c=$code '{ $1=$1"_"c; print $0; }' | \
      ali-to-post ark:- ark,scp:$dir/ali-post/$code.ark,$dir/ali-post/$code.scp
  done
  # pasting the ali's, adding language-specific offsets to the posteriors,
  featlen="ark:feat-to-len 'scp:cat $data_tr90/feats.scp $data_cv10/feats.scp |' ark,t:- |" # get number of frames for every utterance,
  post_scp_list=$(echo ${lang_code[@]} | tr ' ' '\n' | awk -v d=$dir '{ printf(" scp:%s/ali-post/%s.scp", d, $1); }')
  paste-post --allow-partial=true "$featlen" "${ali_dim_csl}" ${post_scp_list} \
    ark,scp:$dir/ali-post/combined.ark,$dir/ali-post/combined.scp
fi

# Train the <BlockSoftmax> system, 1st stage of Stacked-Bottleneck-Network,
if [ $stage -le 2 ]; then  
  case $nnet_type in
    bn)
    # Bottleneck network (40 dimensional bottleneck is good for fMLLR),
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 2 --hid-dim 1500 --bn-dim 40 \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --feat-type "traps" --splice 5 --traps-dct-basis 6 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    sbn)
    # Stacked Bottleneck Netowork, no fMLLR in between,
    bn1_dim=80
    bn2_dim=30
    # Train 1st part,
    dir_part1=${dir}_part1
    $cuda_cmd ${dir}_part1/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 2 --hid-dim 1500 --bn-dim $bn1_dim \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --feat-type "traps" --splice 5 --traps-dct-basis 6 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir_part1
    # Compose feature_transform for 2nd part,
    nnet-initialize <(echo "<Splice> <InputDim> $bn1_dim <OutputDim> $((13*bn1_dim)) <BuildVector> -10 -5:5 10 </BuildVector>") \
      $dir_part1/splice_for_bottleneck.nnet 
    nnet-concat $dir_part1/final.feature_transform "nnet-copy --remove-last-components=4 $dir_part1/final.nnet - |" \
      $dir_part1/splice_for_bottleneck.nnet $dir_part1/final.feature_transform.part1
    # Train 2nd part,
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --feature-transform $dir_part1/final.feature_transform.part1 \
        --hid-layers 2 --hid-dim 1500 --bn-dim $bn2_dim \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    dnn_small)
    # 4 hidden layers, 1024 sigmoid neurons,  
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --cmvn-opts "--norm-means=true --norm-vars=true" \
        --delta-opts "--delta-order=2" --splice 5 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    dnn)
    # 6 hidden layers, 2048 simgoid neurons,
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 6 --hid-dim 2048 \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --delta-opts "--delta-order=2" --splice 5 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    *)
    echo "Unknown --nnet-type $nnet_type"; exit 1;
    ;;
  esac
fi

exit 0

