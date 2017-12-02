#!/bin/bash

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

extract_feats=true

. ./utils/parse_options.sh

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
#set -u           #Fail on an undefined variable

lexicon=data/local/lexicon.txt
if $extend_lexicon; then
  lexicon=data/local/lexiconp.txt
fi

./local/check_tools.sh || exit 1

#Preparing dev2h and train directories
if [ ! -f data/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./data/raw_train_data
    train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`
    touch data/raw_train_data/.done
fi
nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`

mkdir -p data/local
if [[ ! -f $lexicon || $lexicon -ot "$lexicon_file" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------
  local/make_lexicon_subset.sh $train_data_dir/transcription $lexicon_file data/local/filtered_lexicon.txt
  local/prepare_lexicon.pl  --phonemap "$phoneme_mapping" \
    $lexiconFlags data/local/filtered_lexicon.txt data/local
fi

mkdir -p data/lang
if [[ ! -f data/lang/L.fst || data/lang/L.fst -ot $lexicon ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in data/lang on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh \
    --share-silence-phones true \
    data/local $oovSymbol data/local/tmp.lang data/lang
fi

if [[ ! -f data/train/wav.scp || data/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic training lists in data/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/train
  local/prepare_acoustic_training_data.pl \
    --vocab $lexicon --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log
fi

echo ---------------------------------------------------------------------
echo "Starting plp feature extraction for data/train in plp on" `date`
echo ---------------------------------------------------------------------

if [ $extra_feats ] && [ ! -f data/train/.plp.done ]; then
  steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp_pitch/train plp
  utils/fix_data_dir.sh data/train
  steps/compute_cmvn_stats.sh data/train exp/make_plp_pitch/train plp
  utils/fix_data_dir.sh data/train
  touch data/train/.plp.done
fi
