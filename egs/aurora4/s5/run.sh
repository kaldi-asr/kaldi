#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

stage=0
train_set=multi # Set this to 'clean' or 'multi'
test_sets="test_eval92"
train=true   # set to false to disable the training-related scripts
             # note: you probably only want to set --train false if you
             # are using at least --stage 1.
decode=true  # set to false to disable the decoding-related scripts.

. utils/parse_options.sh

#clean LDC wsj0 corpus available in CLSP server: /export/corpora5/LDC/LDC93S6B
#aurora4 directory in CLSP server: /export/corpora5/AURORA

#aurora4=/mnt/spdb/aurora4
aurora4=/export/corpora5/AURORA
#we need lm, trans, from WSJ0 CORPUS
#wsj0=/mnt/spdb/wall_street_journal
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

if [ $stage -le 0 ]; then
  local/aurora4_data_prep.sh $aurora4 $wsj0
fi

if [ $stage -le 1 ]; then
  local/wsj_prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang
fi

if [ $stage -le 2 ]; then
  local/aurora4_format_data.sh
fi

mfccdir=mfcc
if [ $stage -le 3 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  for x in train_si84_clean train_si84_multi test_eval92 test_0166 dev_0330 dev_1206; do 
   steps/make_mfcc.sh  --nj 10 \
     data/$x exp/make_mfcc/$x $mfccdir || exit 1;
   steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
fi

model_affix=
if [ $train_set == 'multi' ]; then
  model_affix=_multi
fi

if [ $stage -le 4 ]; then
  # Note: the --boost-silence option should probably be omitted by default
  # for normal setups.  It doesn't always help. [it's to discourage non-silence
  # models from modeling silence.]
  if $train; then
    steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
      data/train_si84_${train_set} data/lang exp/mono0a${model_affix} || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_test_tgpr exp/mono0a${model_affix} exp/mono0a${model_affix}/graph_tgpr && \
    steps/decode.sh --nj 8 --cmd "$decode_cmd" \
      exp/mono0a${model_affix}/graph_tgpr data/test_eval92 exp/mono0a${model_affix}/decode_tgpr_eval92 
  fi
fi

if [ $stage -le 5 ]; then
  # tri1
  if $train; then
    steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
       data/train_si84_${train_set} data/lang exp/mono0a${model_affix} exp/mono0a${model_affix}_ali || exit 1;

    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
        2000 10000 data/train_si84_${train_set} data/lang exp/mono0a${model_affix}_ali exp/tri1${model_affix} || exit 1;
  fi
fi

if [ $stage -le 6 ]; then
  # tri2
  if $train; then 
    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
      data/train_si84_${train_set} data/lang exp/tri1${model_affix} exp/tri1${model_affix}_ali_si84 || exit 1;

    steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 \
      data/train_si84_${train_set} data/lang exp/tri1${model_affix}_ali_si84 exp/tri2a${model_affix} || exit 1;

    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
      data/train_si84_${train_set} data/lang exp/tri2a${model_affix} exp/tri2a${model_affix}_ali_si84 || exit 1;
    
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
       --splice-opts "--left-context=3 --right-context=3" \
       2500 15000 data/train_si84_${train_set} data/lang exp/tri2a${model_affix}_ali_si84 exp/tri2b${model_affix} || exit 1;
  fi
  
  if $decode; then
    utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2b${model_affix} exp/tri2b${model_affix}/graph_tgpr_5k || exit 1;
    steps/decode.sh --nj 8 --cmd "$decode_cmd" \
      exp/tri2b${model_affix}/graph_tgpr_5k data/test_eval92 exp/tri2b${model_affix}/decode_tgpr_5k_eval92 || exit 1;
  fi
fi

if [ $stage -le 7 ]; then
  # From 2b system, train 3b which is LDA + MLLT + SAT.

  # Align tri2b system with all the si84 data.
  if $train; then
		steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
      data/train_si84_${train_set} data/lang exp/tri2b${model_affix} exp/tri2b${model_affix}_ali_si84  || exit 1;
    
    steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
      data/train_si84_${train_set} data/lang exp/tri2b${model_affix}_ali_si84 exp/tri3b${model_affix} || exit 1;
  fi

  if $decode; then
    nspk=$(wc -l <data/test_eval92/spk2utt)
    utils/mkgraph.sh data/lang_test_tgpr \
      exp/tri3b${model_affix} exp/tri3b${model_affix}/graph_tgpr || exit 1;
    steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
      exp/tri3b${model_affix}/graph_tgpr data/test_eval92 exp/tri3b${model_affix}/decode_tgpr_eval92 || exit 1;
  fi
fi

# Chain training
if [ $stage -le 8 ]; then
  # Caution: this part needs a GPU.
  local/chain/run_tdnn.sh 
fi
exit 1
if [ $stage -le 8 ]; then
  # Estimate pronunciation and silence probabilities.

  # Silprob for normal lexicon.
  #steps/get_prons.sh --cmd "$train_cmd" \
  #  data/train_si84_${train_set} data/lang exp/tri3b${model_affix} || exit 1;
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict \
    exp/tri3b${model_affix}/pron_counts_nowb.txt exp/tri3b${model_affix}/sil_counts_nowb.txt \
    exp/tri3b${model_affix}/pron_bigram_counts_nowb.txt data/local/dict || exit 1

  utils/prepare_lang.sh data/local/dict \
    "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

  for lm_suffix in bg bg_5k tg tg_5k tgpr tgpr_5k; do
    mkdir -p data/lang_test_${lm_suffix}
    cp -r data/lang/* data/lang_test_${lm_suffix}/ || exit 1;
    rm -rf data/lang_test_${lm_suffix}/tmp
    cp data/lang_test_${lm_suffix}/G.* data/lang_test_${lm_suffix}/
  done
fi
exit 1
if [ $stage -le 9 ]; then
  # From 3b system, now using data/lang as the lang directory (we have now added
  # pronunciation and silence probabilities), train another SAT system (tri4b).

  if $train; then
    steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
      data/train_si84_${train_set} data/lang exp/tri3b${model_affix} exp/tri4b${model_affix} || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_test_tgpr_5k \
      exp/tri4b${model_affix} exp/tri4b${model_affix}/graph_tgpr_5k || exit 1;

    for data in 0166 eval92; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode_fmllr.sh --nj ${nspk} --cmd "$decode_cmd" \
        exp/tri4b${model_affix}/graph_tgpr_5k data/test_${data} \
        exp/tri4b${model_affix}/decode_tgpr_5k_${data} || exit 1;
    done
  fi
fi


exit 0;

