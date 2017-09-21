#!/bin/bash

stage=0
nj=20
color=1
data_dir=data
exp_dir=exp
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
    local/make_feature_vect.py $data_dir/$f --scale-size 40 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:$data_dir/$f/data/images.ark,$data_dir/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh $data_dir/$f || exit 1;
  done

fi

numSilStates=4
numStates=8
num_gauss=10000
numLeavesTri=500
numGaussTri=20000

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $data_dir/train/ $data_dir/test/ $data_dir/train/dict
  utils/prepare_lang.sh --num-sil-states $numSilStates --num-nonsil-states $numStates --position-dependent-phones false \
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
  
  # cp -R $data_dir/lang -T $data_dir/lang_test_corpus
  # gunzip -k data/local/local_lm/data/arpa/3gram_big.arpa.gz
  # local/prepare_lm.sh data/local/local_lm/data/arpa/3gram_big.arpa $data_dir/lang_test_corpus || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj \
    $data_dir/train \
    $data_dir/lang \
    $exp_dir/mono
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test \
    $exp_dir/mono \
    $exp_dir/mono/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/mono/graph \
    $data_dir/test \
    $exp_dir/mono/decode_test
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj \
    $data_dir/train $data_dir/lang \
    $exp_dir/mono \
    $exp_dir/mono_ali
  steps/train_deltas.sh \
    $numLeavesTri $numGaussTri $data_dir/train $data_dir/lang \
    $exp_dir/mono_ali \
    $exp_dir/tri
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh $data_dir/lang_test \
    $exp_dir/tri \
    $exp_dir/tri/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/tri/graph \
    $data_dir/test \
    $exp_dir/tri/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    $data_dir/train $data_dir/lang \
    $exp_dir/mono \
    $exp_dir/mono_ali
  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesTri $numGaussTri \
    $data_dir/train $data_dir/lang \
    $exp_dir/mono_ali $exp_dir/tri2
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh $data_dir/lang_test \
    $exp_dir/tri2 \
    $exp_dir/tri2/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/tri2/graph \
    $data_dir/test \
    $exp_dir/tri2/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd \
    --use-graphs true \
    $data_dir/train $data_dir/lang \
    $exp_dir/tri2 \
    $exp_dir/tri2_ali
  steps/train_sat.sh --cmd $cmd \
    $numLeavesTri $numGaussTri \
    $data_dir/train $data_dir/lang \
    $exp_dir/tri2_ali $exp_dir/tri3
fi

if [ $stage -le 11 ]; then
  utils/mkgraph.sh $data_dir/lang_test \
    $exp_dir/tri3 \
    $exp_dir/tri3/graph
  steps/decode_fmllr.sh --nj $nj --cmd $cmd \
    $exp_dir/tri3/graph \
    $data_dir/test \
    $exp_dir/tri3/decode_test
fi

if [ $stage -le 12 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd \
    --use-graphs true \
    $data_dir/train $data_dir/lang \
    $exp_dir/tri3 \
    $exp_dir/tri3_ali
fi

affix=1a
affix_chain=1b_chainali
nnet3_affix=fsf4
common_egs_dir=$exp_dir/chainfsf4/cnn1a_1/egs

if [ $stage -le 13 ]; then
  local/chain/run_cnn_1a.sh --stage 0 \
   --gmm tri3 \
   --ali tri3_ali \
   --nnet3_affix $nnet3_affix \
   --affix $affix \
   --lang_test lang_test
fi

if [ $stage -le 14 ]; then
  local/chain/run_cnn_chainali_1a.sh --stage 0 \
   --gmm tri3 \
   --ali tri3_ali \
   --nnet3_affix $nnet3_affix \
   --affix $affix_chain \
   --chain_model_dir $exp_dir/chain${nnet3_affix}/cnn${affix} \
   --lang_test lang_test
fi
