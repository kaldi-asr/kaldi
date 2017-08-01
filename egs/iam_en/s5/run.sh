#!/bin/bash

stage=0
nj=20
variance_floor_val=0.1
data_dir=char_data
exp_dir=char_exp
totgauss=1000
boost_sil=1.25
numLeavesTri=10
numGaussTri=20
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --nj $nj
fi

mkdir -p $data_dir/{train,val_1,val_2,test}/data
if [ $stage -le 1 ]; then
  for f in train val_1 val_2 test; do
    local/make_feature_vect.py $data_dir/$f --scale-size 40 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:$data_dir/$f/data/images.ark,$data_dir/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh $data_dir/$f || exit 1;
  done

fi

if [ $stage -le 2 ]; then
  local/prepare_char_dict.sh $data_dir/train/ $data_dir/dict
  
  utils/prepare_lang.sh --num-nonsil-states 5 --position-dependent-phones false \
    --sil-prob 0.0 $data_dir/dict "<sil>" $data_dir/lang_5states/temp $data_dir/lang_5states
fi

if [ $stage -le 3 ]; then
  steps/train_mono.sh --nj $nj \
    --variance_floor_val $variance_floor_val --totgauss $totgauss \
    $data_dir/train $data_dir/lang_5states $exp_dir/mono_5states_700
fi

if [ $stage -le 4 ]; then
  cp -R $data_dir/lang_5states -T $data_dir/lang_test_5states
  local/prepare_lm.sh $data_dir/train/text $data_dir/lang_test_5states 2 || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test_5states $exp_dir/mono_5states_700 \
    $exp_dir/mono_5states_700/graph

  steps/decode.sh --nj $nj --cmd $cmd $exp_dir/mono_5states_700/graph $data_dir/test \
    $exp_dir/mono_5states_700/decode_train
fi

if [ $stage -le 4 ]; then

    steps/align_si.sh --boost-silence "$boost_sil" --nj $nj \
     $traindata $lang $exp_dir/mono_grey_1000_3 $exp_dir/mono_ali_1000_3

    steps/train_deltas.sh --variance_floor_val $variance_floor_val \
     $numLeavesTri $numGaussTri $data_dir/train $data_dir/lang_5states $exp_dir/mono_ali_1000_3 $exp_dir/tri_1000_3

fi
