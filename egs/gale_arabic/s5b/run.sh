#!/bin/bash -e

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

num_jobs=120
num_decode_jobs=40
decode_gmm=false
stage=0
overwrite=false
#NB: You can add whatever number of copora you like. The supported extensions 
#NB: (formats) are wav and flac. Flac will be converted using sox and in contrast
#NB: with the old approach, the conversion will be on-the-fly and one-time-only
#NB: during the parametrization.

#NB: Text corpora scpecification. We support either tgz files, which are unpacked
#NB: or just plain (already unpacked) directories. The list of transcript is then
#NB: obtained using find command

#Make sure you edit this section to reflect whers you keep the LDC data on your cluster

#This is CLSP configuration. We add the 2014 GALE data. We got around 2 % 
#improvement just by including it. The gain might be large if someone would tweak
# the number of leaves and states and so on.

audio=(
  /export/corpora/LDC/LDC2013S02/
  /export/corpora/LDC/LDC2013S07/
  /export/corpora/LDC/LDC2014S07/
)
text=(
  /export/corpora/LDC/LDC2013T17
  /export/corpora/LDC/LDC2013T04
  /export/corpora/LDC/LDC2014T17
)

#audio=(
#  /data/sls/scratch/amali/data/GALE/LDC2013S02
#  /data/sls/scratch/amali/data/GALE/LDC2013S07
#  /data/sls/scratch/amali/data/GALE/LDC2014S07
#)
#text=(
#  /data/sls/scratch/amali/data/GALE/LDC2013T17.tgz
#  /data/sls/scratch/amali/data/GALE/LDC2013T04.tgz
#  /data/sls/scratch/amali/data/GALE/LDC2014T17.tgz
#)

galeData=GALE
#prepare the data
#split train dev test 
#prepare lexicon and LM 

# You can run the script from here automatically, but it is recommended to run the data preparation,
# and features extraction manually and and only once.
# By copying and pasting into your shell.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.

if [ $stage -le 0 ]; then

  if [ -f data/train/text ] && ! $overwrite; then
    echo "$0: Not processing, probably script have run from wrong stage"
    echo "Exiting with status 1 to avoid data corruption"
    exit 1;
  fi

  echo "$0: Preparing data..."
  #copy the audio files to local folder wav and convet flac files to wav
  local/gale_data_prep_audio.sh  "${audio[@]}" $galeData || exit 1;
  
  #get the transcription and remove empty prompts and all noise markers  
  local/gale_data_prep_txt.sh  "${text[@]}" $galeData || exit 1;
  
  # split the data to reports and conversational and for each class will have train/dev and test
  local/gale_data_prep_split.sh $galeData  || exit 1;

fi

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
if [ $stage -le 1 ]; then
  echo "$0: Preparing the test and train feature files..."
  for x in train test ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $num_jobs \
      data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x # some files fail to get mfcc for many reasons
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing BPE..."
  cut -d' ' -f2- data/train/text | \
    utils/lang/bpe/prepend_words.py | \
    utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt

  for set in test train; do
    cut -d' ' -f1 data/$set/text > data/$set/ids
    cut -d' ' -f2- data/$set/text | \
      utils/lang/bpe/prepend_words.py | \
      utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt \
      | sed 's/@@//g' > data/$set/bpe_text

    mv data/$set/text data/$set/text.old
    paste -d' ' data/$set/ids data/$set/bpe_text > data/$set/text
    rm -f data/$set/bpe_text data/$set/ids
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: Preparing dictionary and lang..."
  echo "$0: Estimating a language model for decoding..."

  # get all Arabic grapheme dictionaries and add silence and UNK
  local/prepare_dict.sh  || exit 1;

  #prepare the langauge resources
  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false \
    data/local/dict "<sil>" data/local/lang data/lang   || exit 1;

  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang
fi


if [ $stage -le 4 ]; then        
  # LM training
  #local/prepare_lm.sh || exit 1;
  local/train_lm.sh || exit 1;
  #local/gale_format_data.sh  || exit 1;
  # G compilation, check LG composition
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/6gram_unpruned.arpa.gz \
                     data/local/dict/lexicon.txt data/lang

  #utils/format_lm.sh data/lang data/local/lm3.gz data/local/dict/lexicon.txt data/lang/
fi
exit
# Here we start the AM
if [ $stage -le 3 ]; then
  # Let's create a subset with 10k segments to make quick flat-start training:
  utils/subset_data_dir.sh data/train 10000 data/train.10K || exit 1;
fi

if [ $stage -le 5 ]; then
  # Train monophone models on a subset of the data, 10K segment
  # Note: the --boost-silence option should probably be omitted by default
  steps/train_mono.sh --nj 40 --cmd "$train_cmd" \
    data/train.10K data/lang exp/mono || exit 1;
fi

if [ $stage -le 6 ]; then
  # Get alignments from monophone system.
  steps/align_si.sh --nj $num_jobs --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali || exit 1;
  
  # train tri1 [first triphone pass]
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;
fi

if [ $stage -le 7 ] && $decode_gmm; then
  # First triphone decoding
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
  steps/decode.sh  --nj $num_decode_jobs --cmd "$decode_cmd" \
    exp/tri1/graph data/test exp/tri1/decode
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $num_jobs --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

  # train and decode tri2b [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" 4000 50000 \
    data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
fi

if [ $stage -le 9 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
  steps/decode.sh --nj $num_decode_jobs --cmd "$decode_cmd" \
    exp/tri2b/graph data/test exp/tri2b/decode
fi

if [ $stage -le 10 ]; then
  # Align all data with LDA+MLLT system (tri2b)
  steps/align_si.sh --nj $num_jobs --cmd "$train_cmd" \
    --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;
fi

if [ $stage -le 11 ]; then
  local/chain/run_tdnn.sh      #tdnn recipe:
fi

## nnet3 cross-entropy 
#local/nnet3/run_tdnn.sh #tdnn recipe:
#local/nnet3/run_lstm.sh --stage 12  #lstm recipe (we skip ivector training)
#
## chain lattice-free 
#local/chain/run_tdnn.sh      #tdnn recipe:
#local/chain/run_tdnn_lstm.sh #tdnn-lstm recipe:

#if [ $stage -le 15 ]; then
#  time=$(date +"%Y-%m-%d-%H-%M-%S")
#
#  #get detailed WER; reports, conversational and combined
#  local/split_wer.sh $galeData > RESULTS.details.$USER.$time # to make sure you keep the results timed and owned
#
#  echo training succedded
#fi

exit 0

#TODO:
#LM (4-gram and RNN) rescoring
#combine lattices
#dialect detection
