#!/bin/bash

# Copyright 2015  University of Illinois (Author: Amit Das)
# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)

# Apache 2.0

# This example script trains Multi-lingual DNN with <BlockSoftmax> output, using FBANK features.
# The network is trained on multiple languages simultaneously, creating a separate softmax layer
# per language while sharing hidden layers across all languages.
# The script supports arbitrary number of languages.

. path.sh
. cmd.sh

echo
echo "## LOG: $0 $@"
echo
# begin options
cmd=run.pl
nj=40
steps=
validating_rate=0.1
pretrain_rate=0.2
learn_rate=0.008
train_tool=nnet-train-frmshuff-mling
train_tool_opts="--minibatch-size=2048 --randomizer-size=32768 --randomizer-seed=777"
train_opts="--nn-depth 6 --hid-dim 2048 --splice 10"
pretrain_cmd="steps/nnet/pretrain_dbn.sh --feat-type traps  --copy_feats_tmproot /local/hhx502"
dnn_nnet_init=
train_cmd="steps/nnet/train.sh --copy_feats_tmproot /local/hhx502"
cmvn_opts="--norm-means=true --norm-vars=true"
delta_opts="--delta-order=2"

bn1_dim=80
train_part1_cmd="steps/nnet/train.sh --hid-dim 1500 --hid-layers 2 --feat-type traps --splice 5"
bnfe_nnet_init=
bn2_dim=30
train_part2_cmd="steps/nnet/train.sh --hid-layers 2 --hid-dim 1500"

# end options

. parse_options.sh  || exit 1

function Usage {
 cat<<END

 Usage $(basename $0) [options] <lang_code_csl> <lang_weight_csl> <ali_dir_csl> <data_dir_csl> <tgtdir>
 [options]:
 --cmd                                  # value, "$cmd"
 --steps                                # value, "$steps"
 --validating-rate                      # value, "$validating_rate"
 --pretrain-rate                        # value, $pretrain_rate
 --learn-rate                           # value, $learn_rate
 --train-tool-opts                      # value, "$train_tool_opts"
 --train-opts                           # value, "$train_opts"
 --pretrain-cmd                         # value, "$pretrain_cmd"
 --dnn-nnet-init                        # value, "$dnn_nnet_init"
 --train-cmd                            # value, "$train_cmd"
 --cmvn-opts                            # value, "$cmvn_opts"
 --delta-opts                           # value, "$delta_opts"

 --bn1-dim                              # value, $bn1_dim
 --train-part1-cmd                      # value, "$train_part1_cmd"
 --bnfe-nnet-init                       # value, "$bnfe_nnet_init"
 --bn2-dim                              # value, $bn2_dim
 --train-part2-cmd                      # value, "$train_part2_cmd"
 [steps]: 
 1: prepare features
 2: prepare targets
 3: pretraining
 4: dnn training
 5: train part1 bnf extractor
 6: prepare feature_transform for part2 bnf extractor
 7: train part2 bnf extractor
 
 8: make target label (splitting method)
 9: make mling_opts_csl for dnn (go back to step3 for pretraining)
 10: train mling dnn (splitting method)
 
 11: make mling_opts_csl for bnfe1
 12: train bnfe1 (go back to step6)
 13: make mling_opts_csl for bnfe2
 14: train bnfe2

 [example]:
 
 $0 --steps 4  cant,assa,beng,pashto \
     1.0,1.0,1.0,1.0 \
     /home2/hhx502/kws2016/babel101b-v0.4c-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel102b-v0.5a-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel103b-v0.4b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel104b-v0.4bY-build/exp/tri4a/ali_merge-train \
    /home2/hhx502/kws2016/babel101b-v0.4c-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel102b-v0.5a-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel103b-v0.4b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel104b-v0.4bY-build/data/merge_train/fbank-pitch   \
 /home2/hhx502/kws2016/mling4-test

$0 --steps 1,8,3,9,10 cant,pash,turk,taga,viet 1.0,1.0,1.0,1.0,1.0 \
../w2015/monoling/cant101/llp2/exp/tri4a/ali_train,../w2015/monoling/pash104/llp2/exp/tri4a/ali_train,../w2015/monoling/turk105/llp2/exp/tri4a/ali_train,../w2015/monoling/taga106/llp2/exp/tri4a/ali_train,../w2015/monoling/viet107/llp2/exp/tri4a/ali_train \
../w2015/monoling/cant101/llp2/data/train/fbank-pitch,../w2015/monoling/pash104/llp2/data/train/fbank-pitch,../w2015/monoling/turk105/llp2/data/train/fbank-pitch,../w2015/monoling/taga106/llp2/data/train/fbank-pitch,../w2015/monoling/viet107/llp2/data/train/fbank-pitch \
kws2016/llp2-mling/exp/mling5-test

$0 --steps 1,2,3,4  cant,assa,beng,pash,turk,taga,viet,hait,swah,lao,tami,kurm,zulu,tokp,cebu,kaza,telu,lith,guar,igbu,amha,mong,java,dhol \
1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 \
/home2/hhx502/kws2016/babel101b-v0.4c-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel102b-v0.5a-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel103b-v0.4b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel104b-v0.4bY-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel105b-v0.5-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel106b-v0.2g-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel107b-v0.7-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel201b-v0.2b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel202b-v1.0d-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel203b-v3.1a-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel204b-v1.1b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel205b-v1.0a-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel206b-v0.1e-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel207b-v1.0e-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel301b-v2.0b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel302b-v1.0a-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel303b-v1.0a-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel304b-v1.0b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel305b-v1.0c-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel306b-v2.0c-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel307b-v1.0b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel401b-v2.0b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel402b-v1.0b-build/exp/tri4a/ali_merge-train,/home2/hhx502/kws2016/babel403b-v1.0b-build/exp/tri4a/ali_merge-train \
/home2/hhx502/kws2016/babel101b-v0.4c-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel102b-v0.5a-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel103b-v0.4b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel104b-v0.4bY-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel105b-v0.5-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel106b-v0.2g-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel107b-v0.7-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel201b-v0.2b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel202b-v1.0d-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel203b-v3.1a-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel204b-v1.1b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel205b-v1.0a-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel206b-v0.1e-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel207b-v1.0e-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel301b-v2.0b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel302b-v1.0a-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel303b-v1.0a-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel304b-v1.0b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel305b-v1.0c-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel306b-v2.0c-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel307b-v1.0b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel401b-v2.0b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel402b-v1.0b-build/data/merge_train/fbank-pitch,/home2/hhx502/kws2016/babel403b-v1.0b-build/data/merge_train/fbank-pitch \
/home2/hhx502/kws2016/mling24

END
}

if [ $# -ne 5 ]; then
  echo "## lOG: $0 $@"
  Usage && exit 1
fi
lang_code_csl=$1
lang_weight_csl=$2
ali_dir_csl=$3
data_dir_csl=$4
tgtdir=$5

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    declare step$index=1
  done
fi

## set -euxo pipefail
pretrain_cmd="$pretrain_cmd $train_opts"
[ ! -z $dnn_nnet_init ] && pretrain_cmd="$pretrain_cmd --only-feature-transform true"
train_cmd="$train_cmd --learn-rate $learn_rate"
train_part1_cmd="$train_part1_cmd --learn-rate $learn_rate --bn-dim $bn1_dim"
train_part2_cmd="$train_part2_cmd --learn-rate $learn_rate --bn-dim $bn2_dim"
# Convert 'csl' to bash array (accept separators ',' ':'),
lang_code=($(echo $lang_code_csl | tr ',:' ' ')) 
ali_dir=($(echo $ali_dir_csl | tr ',:' ' '))
data_dir=($(echo $data_dir_csl | tr ',:' ' '))

# Make sure we have same number of items in lists,
! [ ${#lang_code[@]} -eq ${#ali_dir[@]} -a ${#lang_code[@]} -eq ${#data_dir[@]} ] && \
  echo "## ERROR, Non-matching number of 'csl' items: lang_code ${#lang_code[@]}, ali_dir ${ali_dir[@]}, data_dir ${#data_dir[@]}" && \
  exit 1
num_langs=${#lang_code[@]}
# Check if all the input directories exist,
for i in $(seq 0 $[num_langs-1]); do
  echo "lang = ${lang_code[$i]}, alidir = ${ali_dir[$i]}, datadir = ${data_dir[$i]}"
  [ ! -d ${ali_dir[$i]} ] && echo  "Missing ${ali_dir[$i]}" && exit 1
  [ ! -d ${data_dir[$i]} ] && echo "Missing ${data_dir[$i]}" && exit 1
done
data=$tgtdir/data-resource
train_data=$data/combined-tr$validating_rate
cv_data=$data/combined-cv$validating_rate
pretrain_data=$data/combined-pretrain$pretrain_rate
if [ ! -z $step01 ]; then
  echo "## LOG: prepare data started @ `date`"
  train_x=""
  cv_x=""
  for i in $(seq 0 $[num_langs-1]);do
    code=${lang_code[$i]}
    sdata1=${data_dir[$i]}
    sdata=$data/$code
    utils/copy_data_dir.sh --utt-prefix ${code}_ --spk-prefix ${code}_ $sdata1 $sdata
    cur_tr=$data/${code}_train
    cur_cv=$data/${code}_cv
    echo "sdata=$sdata, cur_tr=$cur_tr, cur_cv=$cur_cv"
    local/nnet/subset_data.sh --subset_time_ratio $validating_rate \
    --random true \
    --data2 $cur_tr \
    $sdata  $cur_cv || exit 1
    cat $cur_tr/utt2spk | awk -v c=$code '{$2=c; print;}' > $cur_tr/utt2lang
    cat $cur_cv/utt2spk | awk -v c=$code '{$2=c; print;}' > $cur_cv/utt2lang
    train_x="$train_x $cur_tr"
    cv_x="$cv_x $cur_cv"
  done
  # Merge the datasets
  utils/combine_data.sh $train_data $train_x
  utils/combine_data.sh $cv_data $cv_x
  # Validate
  utils/validate_data_dir.sh $train_data
  utils/validate_data_dir.sh $cv_data
  local/nnet/subset_data.sh --subset_time_ratio $pretrain_rate \
  --random true \
  $train_data  $pretrain_data || exit 1
  echo "## LOG: data preparation done @ `date`"
fi

# Extract the tied-state numbers from transition models,
for i in $(seq 0 $[num_langs-1]); do
  ali_dim[i]=$(hmm-info ${ali_dir[i]}/final.mdl | grep pdfs | awk '{ print $NF }')
done
ali_dim_csl=$(echo ${ali_dim[@]} | tr ' ' ',')
echo "## LOG: ali_dim_csl=$ali_dim_csl"

# Total number of DNN outputs (sum of all per-language blocks),
output_dim=$(echo ${ali_dim[@]} | tr ' ' '\n' | awk '{ sum += $i; } END{ print sum; }')
echo "## LOG: Total number of DNN outputs: $output_dim = $(echo ${ali_dim[@]} | sed 's: : + :g')"

# Objective function string (per-language weights are imported from '$lang_weight_csl'),
objective_function="multitask$(echo ${ali_dim[@]} | tr ' ' '\n' | \
  awk -v w=$lang_weight_csl 'BEGIN{ split(w,w_arr,/[,:]/); } { printf(",xent,%d,%s", $1, w_arr[NR]); }')"
echo "## LOG: Multitask objective function: $objective_function"

tgtalidir=$data/ali-post
if [ ! -z $step02 ]; then
  echo "## LOG: prepare to merge the targets started @ `date`"
  [ -d $tgtalidir ] || mkdir -p $tgtalidir
  # re-saving the ali in posterior format, indexed by 'scp',
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[$i]}
    ali=${ali_dir[$i]}
    # utt suffix added by 'awk',
    ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | awk -v c=$code '{ $1=c"_"$1; print $0; }' | \
    ali-to-post ark:- ark,scp:$tgtalidir/$code.ark,$tgtalidir/$code.scp
  done
  # pasting the ali's, adding language-specific offsets to the posteriors,
  featlen="ark:feat-to-len 'scp:cat $train_data/feats.scp $cv_data/feats.scp |' ark,t:- |" # get number of frames for every utterance,
  post_scp_list=$(echo ${lang_code[@]} | tr ' ' '\n' | awk -v d=$tgtalidir '{ printf(" scp:%s/%s.scp", d, $1); }')
  paste-post --allow-partial=true "$featlen" "${ali_dim_csl}" ${post_scp_list} \
    ark,scp:$tgtalidir/combined.ark,$tgtalidir/combined.scp
  echo "## LOG: targets merging ended @ `date`"
fi
if [ ! -z $step08 ]; then
  echo "## LOG: step08, combine post @ `date`"
  [ -d $tgtalidir ] || mkdir -p $tgtalidir
  # re-saving the ali in posterior format, indexed by 'scp',
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[$i]}
    ali=${ali_dir[$i]}
    # utt suffix added by 'awk',
    ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | awk -v c=$code '{ $1=c"_"$1; print $0; }' | \
    ali-to-post ark:- ark,scp:$tgtalidir/$code.ark,$tgtalidir/$code.scp
  done
  # pasting the ali's, adding language-specific offsets to the posteriors,
  featlen="ark:feat-to-len 'scp:cat $train_data/feats.scp $cv_data/feats.scp |' ark,t:- |" # get number of frames for every utterance,
  post_scp_list=$(echo ${lang_code[@]} | tr ' ' '\n' | awk -v d=$tgtalidir '{ printf(" scp:%s/%s.scp", d, $1); }')
  paste-post --allow-partial=true --no-merge=true "$featlen" "${ali_dim_csl}" ${post_scp_list} \
  ark,scp:$tgtalidir/combined-label.ark,$tgtalidir/combined-label.scp
  echo "## LOG: step08, done @ `date`"
fi
nn_depth=$(echo "$train_opts" | perl -pe 'if(m/--nn-depth\s+(\d+)/){$_=$1;}else{exit 1;}')
hid_dim=$(echo "$train_opts" | perl -pe 'if(m/--hid-dim\s+(\d+)/){$_=$1;}else{exit 1;}')
nnet_dir=$tgtdir/dnn-m${num_langs}-layers$nn_depth
dbn=$tgtdir/pretrain_dbn/${nn_depth}.dbn
feature_transform=$tgtdir/pretrain_dbn/final.feature_transform
nnet_init=$nnet_dir/nnet.init
if [ ! -z $step03 ]; then
  echo "## LOG: pretraining started @ `date`"
  $pretrain_cmd --delta-opts $delta_opts --cmvn-opts "$cmvn_opts" \
  $pretrain_data $tgtdir/pretrain_dbn
  echo "## LOG: done @ `date`"
fi
function make_mling_opts_csl {
  local x_dir=$1
  local x_hid_dim=$2
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[i]}
    local tgt_num=${ali_dim[i]}
    local curdir=$x_dir/$code
    [ -d $curdir ] || mkdir -p $curdir
    utils/nnet/make_nnet_proto.py $x_hid_dim $tgt_num 0 $x_hid_dim > $curdir/nnet.proto
    nnet-initialize $curdir/nnet.proto $curdir/nnet.init
    mnet_dir[i]="$curdir/nnet.init"
  done
  mnet_dir_csl=$(echo "${mnet_dir[*]}" | tr ' ' ',')
  [ -f $x_dir/utt2lang ] || \
  cat $train_data/utt2lang $cv_data/utt2lang > $x_dir/utt2lang
  mling_opts_csl="ark:$x_dir/utt2lang;$lang_code_csl;$mnet_dir_csl"
  echo "$lang_code_csl" >$x_dir/lang_code_csl
  echo "$ali_dir_csl" >$x_dir/ali_dir_csl
  echo "$data_dir_csl" >$x_dir/data_dir_csl
  echo "$ali_dim_csl" >$x_dir/ali_dim_csl
}
if [ ! -z $step09 ]; then
  echo "## LOG: step09, make language dependent softmax nnet @ `date`"
  make_mling_opts_csl $nnet_dir $hid_dim
  x_nnet_init=$nnet_dir/${lang_code[0]}/nnet.init
  [ -f $x_nnet_init ] || { echo "## ERROR: step09, file $x_nnet_init expected"; exit 1; }
  if [ -z $dnn_nnet_init ]; then
    nnet-concat $dbn  $x_nnet_init $nnet_init
  else
    nnet_init=$dnn_nnet_init
  fi
  echo "## LOG: step09, mling_opts_csl=$mling_opts_csl"
  echo "## LOG: step09, done @ `date`"
fi
if [ ! -z $step10 ]; then
  echo "## LOG: step10, train mling dnn @ `date`"
  for x in $nnet_init $feature_transform; do
    [ -e $x ] || { echo "ERROR: step10, $x expected"; exit 1; }
  done
  if [ ! -f $nnet_dir/nnet.init ]; then
    srcnnet=$(cd $(dirname $nnet_init); pwd)/$(basename $nnet_init)
    (cd $nnet_dir; ln -s $srcnnet nnet.init)
  fi
  $train_cmd  --nnet-init $nnet_init  \
        --copy-feats true \
        --feature-transform $feature_transform \
        --train-tool $train_tool \
        --train-tool-opts  "$train_tool_opts" \
        --schedule-cmd "steps/nnet/train_scheduler_mling.sh" \
        --mling-opts "$mling_opts_csl" \
        --labels "scp:$tgtalidir/combined-label.scp" \
        ${train_data} ${cv_data} lang-dummy ali-dummy ali-dummy $nnet_dir

  echo "## LOG: step10, ended @ `date`"
fi
if [ ! -z $step04 ]; then
  echo "## LOG: train multilingual dnn started @ `date`"
   $train_cmd  --hid-dim $hid_dim  \
        --feature-transform $feature_transform \
        --train-tool-opts  "$train_tool_opts" \
        --hid-layers 0 --dbn $dbn \
        --labels "scp:$tgtalidir/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${train_data} ${cv_data} lang-dummy ali-dummy ali-dummy $nnet_dir
  echo "## LOG: ended @ `date`"
fi
dir_part1=$tgtdir/bnf-extractor-part1
if [ ! -z $step05 ]; then
  echo "## LOG: train 1st bnf extractor started @ `date`"
  $train_part1_cmd --cmvn-opts "$cmvn_opts" --delta-opts $delta_opts \
  --train-tool-opts "$train_tool_opts" \
  --labels "scp:$tgtalidir/combined.scp" --num-tgt $output_dim \
  --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
  --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
  ${train_data} ${cv_data} lang-dummy ali-dummy ali-dummy $dir_part1
  echo "## LOG: train 1st ended @ `date`"
fi

if [ ! -z $step11 ]; then
  echo "## LOG: step11, make language dependent softmax nnet (bnf) @ `date`"
  bnfe_hid_dim=$(echo "$train_part1_cmd" | perl -pe 'chomp; if(/--hid-dim\s+(\d+)\s+/) {$_=$1;}')
  make_mling_opts_csl $dir_part1 $bnfe_hid_dim
  echo "## LOG: step11, mling_opts_csl=$mling_opts_csl done @ `date`"
fi
if [ ! -z $step12 ]; then
  echo "## LOG: step12, train 1st bnf extractor @ `date`"
  if [ ! -f $dir_part1/nnet.init ] && [ ! -z $bnfe_nnet_init ]; then
    srcnnet=$(cd $(dirname $bnfe_nnet_init); pwd)/$(basename $bnfe_nnet_init)
    (cd $dir_part1; ln -s $srcnnet nnet.init)
    bnfe_nnet_init=$dir_part1/nnet.init
  fi
  $train_part1_cmd --cmvn-opts "$cmvn_opts" --delta-opts $delta_opts \
  --schedule-cmd "steps/nnet/train_scheduler_mling.sh" \
  --mling-opts "$mling_opts_csl" \
  --train-tool-opts "$train_tool_opts" \
  ${bnfe_nnet_init:+ --nnet-init $bnfe_nnet_init} \
  --labels "scp:$tgtalidir/combined-label.scp" --no-hmm-info true --num-tgt ${ali_dim[0]} \
  --train-tool $train_tool\
  ${train_data} ${cv_data} lang-dummy ali-dummy ali-dummy $dir_part1

  echo "## LOG: step12, done @ `date`"
fi
feature_transform=$dir_part1/final.feature_transform.part1
if [ ! -z $step06 ]; then
  echo "## LOG: prepare feature transform for 2nd part @ `date`"
  # Compose feature_transform for 2nd part,
  nnet-initialize <(echo "<Splice> <InputDim> $bn1_dim <OutputDim> $((13*bn1_dim)) <BuildVector> -10 -5:5 10 </BuildVector>") \
  $dir_part1/splice_for_bottleneck.nnet 
  nnet-concat $dir_part1/final.feature_transform "nnet-copy --remove-last-layers=4 $dir_part1/final.nnet - |" \
  $dir_part1/splice_for_bottleneck.nnet $feature_transform
  echo "## LOG: done @ `date`"
fi
dir_part2=$tgtdir/bnf-extractor-part2
if [ ! -z $step07 ]; then
  echo "## LOG: train 2nd bnf extractor @ `date`"
  $train_part2_cmd  \
  --feature-transform $feature_transform \
  --train-tool-opts  "$train_tool_opts" \
  --labels "scp:$tgtalidir/combined.scp" --num-tgt $output_dim \
  --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
  --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
  ${train_data} ${cv_data} lang-dummy ali-dummy ali-dummy $dir_part2
  echo "## LOG: done @ `date`"
fi
if [ ! -z $step13 ]; then
  echo "## LOG: step13, make multi-softmax nnet (bnf) @ `date`"
  bnfe_hid_dim=$(echo "$train_part2_cmd" | perl -pe 'chomp; if(/--hid-dim\s+(\d+)\s+/) {$_=$1;}')
  echo "## LOG: step13, bnfe_hid_dim=$bnfe_hid_dim"
  make_mling_opts_csl $dir_part2 $bnfe_hid_dim
  echo "## LOG: step13, mling_opts_csl=$mling_opts_csl done @ `date`"
fi

if [ ! -z $step14 ]; then
  echo "## LOG: step14, train 2nd bnf extractor @ `date`"
  $train_part2_cmd  \
  --feature-transform $feature_transform \
  --schedule-cmd "steps/nnet/train_scheduler_mling.sh" \
  --mling-opts "$mling_opts_csl" \
  --train-tool-opts  "$train_tool_opts" \
  --labels "scp:$tgtalidir/combined-label.scp" --no-hmm-info true  --num-tgt ${ali_dim[0]} \
  --train-tool "$train_tool" \
  ${train_data} ${cv_data} lang-dummy ali-dummy ali-dummy $dir_part2
  echo "## LOG: step14, done @ `date`"
fi
