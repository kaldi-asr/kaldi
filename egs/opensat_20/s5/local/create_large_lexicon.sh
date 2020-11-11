#!/bin/bash


. ./cmd.sh
. ./path.sh
stage=0
. utils/parse_options.sh

if [ $stage -le 1 ] ; then
  cp -r /export/c05/aarora8/kaldi2/egs/icsisafet/s5/data/local/dict_nosp /export/c05/aarora8/kaldi2/egs/icsisafet/s5/data/local/dict_nosp_3

  cp /export/c05/aarora8/kaldi2/egs/icsisafet/s5/meta_dexp/kws_exp/data/local/dict_nosp_3/lexicon.txt /export/c05/aarora8/kaldi2/egs/icsisafet/s5/data/local/dict_nosp_3/
  echo "stage 1"
fi

if [ $stage -le 2 ] ; then
  echo "stage 1"
  utils/prepare_lang.sh data/local/dict_nosp_3 '<UNK>' data/local/lang_nosp data/lang_nosp_3

  utils/validate_lang.pl data/lang_nosp_3
fi

if [ $stage -le 3 ] ; then
  echo "stage 1"
  # changed local/safet_train_lms_srilm.sh script to use lang_nosp_3 for words.txt
  local/safet_train_lms_srilm.sh \
    --train_text data/train_safet/text --dev_text data/safe_t_dev1_hires/text  \
    data/ data/local/srilm_3
  utils/format_lm.sh  data/lang_nosp_3/ data/local/srilm_3/lm.gz\
    data/local/dict_nosp_3/lexicon.txt  data/lang_nosp_test_3
fi

if [ $stage -le 4 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_nosp_test_3 exp/ihm/chain_1a/tdnn_b_bigger_2_aug exp/ihm/chain_1a/tdnn_b_bigger_2_aug/graph_3
fi

