#!/bin/bash

# Parameters for extended lexicon.
extend_lexicon=true
unk_fraction_boost=1.0
num_sent_gen=12000000
num_prons=1000000

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
#set -u           #Fail on an undefined variable

lexicon=data/local/lexicon.txt
if $extend_lexicon; then
  lexicon=data/local/lexiconp.txt
fi

#Preparing dev2h and train directories
if [ ! -f data/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./data/raw_train_data
    train_data_dir=`readlink -f ./data/raw_train_data`
    touch data/raw_train_data/.done 
fi
nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`readlink -f ./data/raw_train_data`

if [ ! -d data/raw_dev2h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV2H set"
  echo ---------------------------------------------------------------------  
  local/make_corpus_subset.sh "$dev2h_data_dir" "$dev2h_data_list" ./data/raw_dev2h_data || exit 1
fi

if [ ! -d data/raw_dev10h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV10H set"
  echo ---------------------------------------------------------------------  
  local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./data/raw_dev10h_data || exit 1
fi
nj_max=`cat $dev2h_data_list | wc -l`
if [[ "$nj_max" -lt "$decode_nj" ]] ; then
  echo "The maximum reasonable number of jobs is $nj_max -- you have $decode_nj! (The training and decoding process has file-granularity)"
  exit 1
  decode_nj=$nj_max
fi

# Move data/dev2h preparation forward so we can get data/dev2h/text for
# diagnostic purpose when extending the lexicon.
if [[ ! -f data/dev2h/wav.scp || data/dev2h/wav.scp -ot ./data/raw_dev2h_data/audio ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h data lists in data/dev2h on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/dev2h
  local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    `pwd`/data/raw_dev2h_data data/dev2h > data/dev2h/skipped_utts.log || exit 1
fi

if [[ ! -f data/dev2h/glm || data/dev2h/glm -ot "$glmFile" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h stm files in data/dev2h on" `date`
  echo ---------------------------------------------------------------------
  if [ -z $dev2h_stm_file ]; then 
    echo "WARNING: You should define the variable stm_file pointing to the IndusDB stm"
    echo "WARNING: Doing that, it will give you scoring close to the NIST scoring.    "
    local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev2h || exit 1
  else
    local/augment_original_stm.pl $dev2h_stm_file data/dev2h || exit 1
  fi
  [ ! -z $glmFile ] && cp $glmFile data/dev2h/glm

fi

mkdir -p data/local
if [[ ! -f $lexicon || $lexicon -ot "$lexicon_file" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------
  local/make_lexicon_subset.sh $train_data_dir/transcription $lexicon_file data/local/filtered_lexicon.txt
  local/prepare_lexicon.pl  --phonemap "$phoneme_mapping" \
    $lexiconFlags data/local/filtered_lexicon.txt data/local
  if $extend_lexicon; then
    # Extend the original lexicon.
    # Will creates the files data/local/extend/{lexiconp.txt,oov2prob}.
    mv data/local/lexicon.txt  data/local/lexicon_orig.txt
    local/extend_lexicon.sh --cmd "$train_cmd" --cleanup false \
      --num-sent-gen $num_sent_gen --num-prons $num_prons \
      data/local/lexicon_orig.txt data/local/extend data/dev2h/text
    cp data/local/extend/lexiconp.txt data/local/
  fi
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

if [[ ! -f data/srilm/lm.gz || data/srilm/lm.gz -ot data/train/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM language models on" `date`
  echo ---------------------------------------------------------------------
  # If extending the lexicon, use "--words-file data/local/lexicon_orig.txt" so
  # that the LM is trained just on the vocab that appears in the text. Will add
  # in the OOVs later.
  words_file_param=()
  if $extend_lexicon; then
    words_file_param=(--words-file data/local/lexicon_orig.txt)
  fi
  local/train_lms_srilm.sh "${words_file_param[@]}" --dev-text data/dev2h/text \
    --train-text data/train/text data data/srilm
fi

if [[ ! -f data/lang/G.fst || data/lang/G.fst -ot data/srilm/lm.gz ||\
  ( -f data/local/extend/oov2prob &&\
  data/lang/G.fst -ot data/local/extend/oov2prob ) ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
  extend_lexicon_param=()
  if $extend_lexicon; then
    [ -f data/local/extend/original_oov_rates ] || exit 1;
    unk_fraction=`cat data/local/extend/original_oov_rates |\
      grep "token" | awk -v x=$unk_fraction_boost '{print $NF/100.0*x}'`
    extend_lexicon_param=(--cleanup false --unk-fraction $unk_fraction \
      --oov-prob-file data/local/extend/oov2prob)
  fi
  local/arpa2G.sh ${extend_lexicon_param[@]} \
    data/srilm/lm.gz data/lang data/lang
fi
decode_nj=$dev2h_nj

echo ---------------------------------------------------------------------
echo "Starting plp feature extraction for data/train in plp on" `date`
echo ---------------------------------------------------------------------

if [ ! -f data/train/.plp.done ]; then
  if $use_pitch; then
    steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp_pitch/train plp
  else
    steps/make_plp.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp/train plp
  fi
  utils/fix_data_dir.sh data/train
  steps/compute_cmvn_stats.sh data/train exp/make_plp/train plp
  utils/fix_data_dir.sh data/train
  touch data/train/.plp.done
fi

touch data/.extlex

echo -------------------------------------------------------------------------
echo "Extended lexicon finished on" `date`. Now running script run-1-main.sh
echo -------------------------------------------------------------------------
./run-1-main.sh
exit 0
