#!/bin/bash

#Copyright      2017  Chun Chieh Chang
#               2017  Ashish Arora

stage=0
nj=20
color=1
augment=false
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options (stage, nj, color, augment)
                            # if supplied.

if [ $stage -le 0 ]; then
  local/prepare_data.sh --nj $nj --download_dir /export/corpora5/handwriting_ocr/IAM
fi
mkdir -p data/{train,test}/data

if [ $stage -le 1 ]; then
  # process image extract raw pixel features
  local/make_features.py data/test --feat-dim 40 | \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/test/data/images.ark,data/test/feats.scp || exit 1
  steps/compute_cmvn_stats.sh data/test || exit 1;

  if $augment; then
    # create a backup directory to store text, utt2spk and image.scp file
    mkdir -p data/train/backup
    mv data/train/text data/train/utt2spk data/train/images.scp data/train/backup/
    local/augment_and_make_features.py data/train --feat-dim 40 --vertical-shift 10 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:data/train/data/images.ark,data/train/feats.scp || exit 1
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  else
    local/make_features.py data/train --feat-dim 40 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:data/train/data/images.ark,data/train/feats.scp || exit 1
  fi
  steps/compute_cmvn_stats.sh data/train || exit 1;
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/train/ data/test/ data/train/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.95 \
    data/train/dict "<unk>" data/lang/temp data/lang
fi

if [ $stage -le 3 ]; then
  local/iam_train_lm.sh
  mkdir -p data/lang_test
  cp -R data/lang/. data/lang_test/
  gunzip -k -f data/local/local_lm/data/arpa/3gram_big.arpa.gz
  local/prepare_lm.sh data/local/local_lm/data/arpa/3gram_big.arpa data/lang_test || exit 1;

  # prepare the unk model for open-vocab decoding
  utils/lang/make_unk_lm.sh --ngram-order 4 --num-extra-ngrams 7500 data/train/dict exp/unk_lang_model
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 \
                        --unk-fst exp/unk_lang_model/unk_fst.txt data/train/dict "<unk>" data/lang/temp data/lang_unk
  cp data/lang_test/G.fst data/lang_unk/G.fst
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd --totgauss 10000 data/train \
    data/lang exp/mono
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd $cmd 500 20000 data/train data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh data/lang_test exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/tri exp/tri_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train data/lang exp/tri_ali exp/tri2
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri2/graph \
    data/test exp/tri2/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd $cmd 500 20000 \
    data/train data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 11 ]; then
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph

  steps/decode_fmllr.sh --nj $nj --cmd $cmd exp/tri3/graph \
    data/test exp/tri3/decode_test
fi

if [ $stage -le 12 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 13 ]; then
  local/chain/run_cnn_1a.sh --lang-test lang_unk
fi

if [ $stage -le 14 ]; then
  local/chain/run_cnn_chainali_1b.sh --chain-model-dir exp/chain/cnn_1a \
    --lang-test lang_unk
fi
