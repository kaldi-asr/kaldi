#!/bin/bash

stage=0
nj=20
variance_floor_val=0.1
color=3
data_dir=data_20
exp_dir=exp_20
boost_sil=1.25
numLeavesTri=10
numGaussTri=20
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --nj $nj --dir $data_dir
fi

mkdir -p $data_dir/{train,val_1,val_2,test}/data
if [ $stage -le 1 ]; then
  for f in train val_1 val_2 test; do
    local/make_feature_vect.py $data_dir/$f --scale-size 20 --color $color | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:$data_dir/$f/data/images.ark,$data_dir/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh $data_dir/$f || exit 1;
  done

fi

numStates=5
num_states=_5states
num_gauss=700

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $data_dir/train/ $data_dir/dict
  utils/prepare_lang.sh --num-nonsil-states $numStates --position-dependent-phones false \
    $data_dir/dict "<sil>" $data_dir/lang$num_states/temp $data_dir/lang$num_states
fi

if [ $stage -le 3 ]; then
  ## Starting basic training on features
  ## passing value for variance floor
  steps/train_mono.sh --nj $nj --variance_floor_val $variance_floor_val \
    $data_dir/train $data_dir/lang$num_states $exp_dir/mono${num_states}_$num_gauss
fi

if [ $stage -le 4 ]; then
  cp -R $data_dir/lang$num_states -T $data_dir/lang_test$num_states
  local/prepare_lm.sh $data_dir/train/text $data_dir/lang_test$num_states 2 || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test$num_states \
    $exp_dir/mono${num_states}_$num_gauss $exp_dir/mono${num_states}_$num_gauss/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/mono${num_states}_$num_gauss/graph $data_dir/test $exp_dir/mono${num_states}_$num_gauss/decode_train

if [ $stage -le 6 ]; then

  steps/align_si.sh --boost-silence "$boost_sil" --nj $nj \
    $traindata $lang $exp_dir/mono_grey_1000_3 $exp_dir/mono_ali_1000_3

  steps/train_deltas.sh --variance_floor_val $variance_floor_val \
    $numLeavesTri $numGaussTri $data_dir/train $data_dir/lang_5states $exp_dir/mono_ali_1000_3 $exp_dir/tri_1000_3
fi
