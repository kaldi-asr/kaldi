#!/bin/bash

stage=0
nj=20
variance_floor_val=0.1
data_dir=data_gray
exp_dir=exp_gray
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

#  mkdir -p data/train/log
#  image/split_ocr_dir.sh data/train/ $nj
#  $cmd JOB=1:$nj data/train/log/make_feature_vect.JOB.log \
#    local/make_feature_vect.py data/train/split${nj}/JOB/ --scale-size 40 \| \
#    copy-feats --compress=true --compression-method=7 \
#    ark:- ark,scp:data/train/split${nj}/JOB/data/images.ark,data/train/split${nj}/JOB/feats.scp \
#    || exit 1
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $data_dir/train/ $data_dir/dict
  utils/prepare_lang.sh --num-nonsil-states 5 --position-dependent-phones false $data_dir/dict "<sil>" $data_dir/lang_5states/temp $data_dir/lang_5states
fi

if [ $stage -le 3 ]; then
  ## Starting basic training on features
  ## passing value for variance floor
  steps/train_mono.sh --nj $nj \
    --variance_floor_val $variance_floor_val $data_dir/train $data_dir/lang_5states $exp_dir/mono_5states_700
fi

if [ $stage -le 4 ]; then
  cp -R $data_dir/lang_5states -T $data_dir/lang_test_5states
  local/prepare_lm.sh $data_dir/train/text $data_dir/lang_test_5states 2 || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test_5states $exp_dir/mono_5states_700 $exp_dir/mono_5states_700/graph
  steps/decode.sh --nj $nj --cmd $cmd $exp_dir/mono_5states_700/graph $data_dir/test $exp_dir/mono_5states_700/decode_train
fi
