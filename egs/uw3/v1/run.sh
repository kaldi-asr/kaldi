#!/bin/bash

stage=0
nj=30
data_download=data
data_dir=data
exp_dir=exp

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # Data preparation
  local/prepare_data.sh --dir $data_download
fi

mkdir -p $data_dir/{train,test}/data
if [ $stage -le 1 ]; then
  for f in train test; do
    local/make_feature_vect.py --scale-size 40 --color 1 --pad true $data_download/$f | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:$data_dir/$f/data/images.ark,$data_dir/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh $data_dir/$f || exit 1;
  done
fi

numLeavesTri=500
numGaussTri=20000
numLeavesMLLT=500
numGaussMLLT=20000

beam=50

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $data_dir/train/ $data_dir/test/ $data_dir/train/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --position-dependent-phones false \
    $data_dir/train/dict "<sil>" $data_dir/lang/temp $data_dir/lang
fi

if [ $stage -le 3 ]; then
  cp -R $data_dir/lang -T $data_dir/lang_test

  cp $data_dir/train/text $data_dir/train/text_copy
  cat $data_dir/test/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >test_words.txt
  cat $data_dir/train/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >train_words.txt
  filter_scp.pl --exclude train_words.txt test_words.txt >diff.txt
  cat diff.txt | awk '{ print "id " $1 }' >> $data_dir/train/text_copy

  local/prepare_lm.sh $data_dir/train/text_copy $data_dir/lang_test || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/mono
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/mono \
    $exp_dir/mono_ali
  steps/train_deltas.sh --cmd $cmd \
    $numLeavesTri $numGaussTri \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/mono_ali \
    $exp_dir/tri
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/tri \
    $exp_dir/tri_ali
  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesMLLT $numGaussMLLT \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/tri_ali \
    $exp_dir/tri2
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test \
    $exp_dir/mono \
    $exp_dir/mono/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    $exp_dir/mono/graph \
    $data_dir/test \
    $exp_dir/mono/decode_test
fi

if [ $stage -le 8 ]; then
  utils/mkgraph.sh $data_dir/lang_test \
    $exp_dir/tri \
    $exp_dir/tri/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    $exp_dir/tri/graph \
    $data_dir/test \
    $exp_dir/tri/decode_test
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh $data_dir/lang_test \
    $exp_dir/tri2 \
    $exp_dir/tri2/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    $exp_dir/tri2/graph \
    $data_dir/test \
    $exp_dir/tri2/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd --use-graphs true \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/tri2 \
    $exp_dir/tri2_ali
fi

if [ $stage -le 11 ]; then
  run_cnn_1a.sh --stage 0
fi
