#!/usr/bin/env bash
# Copyright 2021 Xiaomi Corporation (Author: Yongqing Wang)
# Apache 2.0

. ./cmd.sh
. ./path.sh

stage=0

gigaspeech_root=~/GigaSpeech_data/
train_sets=train
test_sets="dev test"
lm_dir=data/local/lm
dict_dir=data/local/dict

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

for f in $lm_dir $dict_dir; do
  [ ! -d $f ] && mkdir -p $f
done
g2p_model_dir=$gigaspeech_root/dict/g2p

if [ $stage -le 0 ]; then
  echo -e "======Download GigaSpeech START|current time : `date +%Y-%m-%d-%T`======"
  # Download and analyze GigaSpeech datasets
  # cd GigaSpeech to run toolkits/kaldi/dowload_analyze.sh to download and analyze meta
  git clone https://github.com/SpeechColab/GigaSpeech.git
  pushd GigaSpeech
  gigaspeech_download.sh $gigaspeech_root
  popd
  echo -e "======Download GigaSpeech END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 1 ]; then
  echo -e "======Prepare Data START|current time : `date +%Y-%m-%d-%T`======"
  # Prepare GigaSpeech data
  . GigaSpeech/env_vars.sh
  GigaSpeech/toolkits/kaldi/gigaspeech_data_prep.sh $gigaspeech_root data true
  echo -e "======Prepare Data END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 2 ]; then
  echo -e "======Prepare Dict START|current time : `date +%Y-%m-%d-%T`======"
  # Prepare dict 
  # If the lexicon is downloaded from GigaSpeech, you can skip the G2P steps with "--stage 4" option
  # Otherwise, you should start from "--stage 0" option to use G2P
  [ ! -f $g2p_model_dir/g2p.model.4 ] && echo "$0: Cannot find $g2p_model_dir/g2p.model.4" && exit 1
  if [ -f $dict_dict/lexicon_raw_nosil.txt ]; then
    # If GigaSpeech has been dowloaded, skip G2P steps with "--stage 4"
    local/prepare_dict.sh --stage 4 $g2p_model_dir/g2p.model.4 data/local/dict
  else
    local/prepare_dict.sh --stage 0 --cmd "$train_cmd" --nj 300 --train-set train --test-sets "$test_sets" $g2p_model_dir/g2p.model.4 data/local/dict
  fi
  echo -e "======Prepare dict END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 3 ]; then
  echo -e "======Train lm START|current time : `date +%Y-%m-%d-%T`======"
  # train lm
  sed 's|\t| |' data/train/text | cut -d " " -f 2- >$lm_dir/text_for_lm.txt
  # The default ngram order is 3
  local/lm/train_lm.sh $lm_dir/text_for_lm.txt $lm_dir
  echo -e "======Train lm END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 4 ]; then
  echo -e "======Prepare lang START|current time : `date +%Y-%m-%d-%T`======"
  utils/prepare_lang.sh data/local/dict \
   "<UNK>" data/local/lang_tmp_nosp data/lang
  
  utils/format_lm.sh data/lang $lm_dir/lm_tgram.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;
  echo -e "======Prepare lang START|current time : `date +%Y-%m-%d-%T`======"

fi

if [ $stage -le 5 ]; then
  echo -e "======Extract feat START|current time : `date +%Y-%m-%d-%T`======"
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  tj1-asr-train-dev* ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl \
      /home/storage{{30..36},{40..49}}/data-tmp-TTL20/$(date +%Y%m%d)/$USER/kaldi-data/$(basename $(pwd))/$(hostname -f)_$(date +%Y%m%d_%H%M%S)_$$/storage/ \
      $mfccdir/storage
  fi

  for part in $test_sets train; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $train_nj data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
  echo -e "======Extract feat END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 6 ]; then
  echo -e "======Subset train data START|current time : `date +%Y-%m-%d-%T`======"
  # Make some small data subsets for early system-build stages.  Note, there are 8283k
  # utterances in the train directory which has 10000 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.
  total_num=`wc -l <data/train/utt2spk`
  
  subset_num=100000
  [ $total_num -lt $subset_num ] && subset_num=$total_num
  utils/subset_data_dir.sh --shortest data/train $subset_num data/train_100k

  
  subset_num=$((total_num/32))
  [ 250000 -lt $subset_num ] && subset_num=250000
  utils/subset_data_dir.sh data/train $subset_num data/train_1d32
  
  subset_num=$((total_num/16))
  [ 500000 -lt $subset_num ] && subset_num=500000
  utils/subset_data_dir.sh data/train $subset_num data/train_1d16

  subset_num=$((total_num/8))
  [ 1000000 -lt $subset_num ] && subset_num=1000000
  utils/subset_data_dir.sh data/train $subset_num data/train_1d8

  echo -e "======Subset train data END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 7 ]; then
  echo -e "======Train mono START|current time : `date +%Y-%m-%d-%T`======"
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
                      data/train_100k data/lang exp/mono
  {
    utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] && echo "Decoder mono Error: no such dir data/$part"
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp/mono/graph data/${part} exp/mono/decode_${part}
      cat exp/mono/decode_${part}/wer_* | utils/best_wer.sh | sed "s/^/mono\t/" > exp/mono/decode_${part}/wer.txt
    done
  } &

  echo -e "======Train mono END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 8 ]; then
  echo -e "======Train tri1b START|current time : `date +%Y-%m-%d-%T`======"
  steps/align_si.sh --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
                    data/train_1d32 data/lang exp/mono exp/mono_ali_train_1d32

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_1d32 data/lang exp/mono_ali_train_1d32 exp/tri1b
  echo -e "======Train tri1b END|current time : `date +%Y-%m-%d-%T`======"
  {
    utils/mkgraph.sh data/lang_test exp/tri1b exp/tri1b/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] && echo "Decoder tri1b Error: no such dir data/$part"
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp/tri1b/graph data/${part} exp/tri1b/decode_${part}
      cat exp/tri1b/decode_${part}/wer_* | utils/best_wer.sh | sed "s/^/tri1b\t/" > exp/tri1b/decode_${part}/wer.txt
    done
  } &

fi

if [ $stage -le 9 ]; then
  echo -e "======Train tri2b START|current time : `date +%Y-%m-%d-%T`======"
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
                    data/train_1d16 data/lang exp/tri1b exp/tri1_ali_train_1d16
  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_1d16 data/lang exp/tri1_ali_train_1d16 exp/tri2b
  {
    utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] && echo "Decoder tri2b Error: no such dir data/$part"
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp/tri2b/graph data/${part} exp/tri2b/decode_${part}
      cat exp/tri2b/decode_${part}/wer_* | utils/best_wer.sh | sed "s/^/tri2b\t/" > exp/tri2b/decode_${part}/wer.txt
    done
  } &
  echo -e "======Train tri2b END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 10 ]; then
  echo -e "======Train tri3b START|current time : `date +%Y-%m-%d-%T`======"
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd" --use-graphs true \
                     data/train_1d16 data/lang exp/tri2b exp/tri2_ali_train_1d16
  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_1d16 data/lang exp/tri2_ali_train_1d16 exp/tri3b
  {
    utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] && echo "Decoder tri3b Error: no such dir data/$part"
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp/tri3b/graph data/$part exp/tri3b/decode_${part}
      cat exp/tri3b/decode_${part}/wer_* | utils/best_wer.sh | sed "s/^/tri3b\t/" > exp/tri3b/decode_${part}/wer.txt
    done
  } &
  echo -e "======Train tri3b END|current time : `date +%Y-%m-%d-%T`======"

fi

if [ $stage -le 11 ]; then
  echo -e "======Train tri4b START|current time : `date +%Y-%m-%d-%T`======"
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    data/train_1d8 data/lang exp/tri3b exp/tri3_ali_train_1d8

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train_1d8 data/lang exp/tri3_ali_train_1d8 exp/tri4b
  {
    utils/mkgraph.sh data/lang_test exp/tri4b exp/tri4b/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] && echo "Decoder tri4b Error: no such dir data/$part"
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp/tri4b/graph data/${part} exp/tri4b/decode_${part}
      cat exp/tri4b/decode_${part}/wer_* | utils/best_wer.sh | sed "s/^/tri4b\t/" > exp/tri4b/decode_${part}/wer.txt
    done
  } &
  echo -e "======Train tri4b END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le -12 ]; then
  echo -e "======Clean up START|current time : `date +%Y-%m-%d-%T`======"
  # this does some data-cleaning. The cleaned data should be useful when we add
  # the neural net and chain systems.  (although actually it was pretty clean already.)
  echo "run cleanup!"
  #local/run_cleanup_segmentation.sh
  echo -e "======Clean up END|current time : `date +%Y-%m-%d-%T`======"
fi

nnet='tdnnf'

if [ $stage -le 13 ]; then
  echo -e "======Train chain START|current time : `date +%Y-%m-%d-%T`======"
  if [ $nnet == 'cnntdnnf' ]; then
    local/chain/run_cnn_tdnn.sh \
      --stage 0 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --train_set train \
      --gmm tri4b \
      --test-sets "$test_sets"
  else
    local/chain/run_tdnn.sh \
      --stage 0 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --train_set train \
      --gmm tri4b \
      --test-sets "$test_sets"
  fi
  echo -e "======Train chain END|current time : `date +%Y-%m-%d-%T`======"
fi

echo "$0: Done"
