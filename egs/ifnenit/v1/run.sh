#!/usr/bin/env bash

stage=0
nj=8
. ./path.sh

# ienit_database points to the database path on the JHU grid.
# you can change this to your local directory of the dataset
ienit_database="/export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data"
train_sets="set_a set_b set_c"
test_sets="set_d"

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  echo "data preparation"
  local/ienit_initialize.sh --database_dir $ienit_database \
     --train_sets "$train_sets" --test_sets "$test_sets"

  local/prepare_data.sh --database_dir $ienit_database \
    --train_sets "$train_sets" --test_sets "$test_sets"
fi

if [ $stage -le 1 ]; then
  # dict folder preparation
  echo "dict folder preparation"
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 3 --num-nonsil-states 4 --position-dependent-phones false data/local/dict "<unk>" data/local/lang data/lang
fi

if [ $stage -le 2 ]; then
  # LM preparation
  echo "LM preparation"
  cat data/train/text | cut -d' ' -f2- | utils/make_unigram_grammar.pl | \
    fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt > data/lang/G.fst
fi

if [ $stage -le 3 ]; then
  steps/train_mono.sh --nj $nj data/train data/lang \
    exp/mono
fi

if [ $stage -le 4 ]; then
  utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh 500 20000 data/train data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 6 ]; then
  utils/mkgraph.sh data/lang exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train data/lang exp/mono_ali exp/tri2
fi

if [ $stage -le 8 ]; then
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri2/graph data/test \
    exp/tri2/decode_test
fi

if [ $stage -le 9 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd $cmd 500 20000 \
    data/train data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 10 ]; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

  steps/decode_fmllr.sh --nj $nj --cmd $cmd exp/tri3/graph \
    data/test exp/tri3/decode_test
fi

if [ $stage -le 11 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 12 ]; then
  local/chain/run_cnn_1a.sh 
fi

if [ $stage -le 13 ]; then
  local/chain/run_cnn_chainali_1a.sh --stage 2
fi
