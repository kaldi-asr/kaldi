#!/bin/bash
# Copyright 2020  Delft University of Technology (Siyuan Feng)
# Copyright 2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail

stage=0
stop_stage=500
extract_feat_nj=8
early_train_nj=60
train_nj=100
phone_ngram_order=2
word_ngram_order=3
# When phone_tokens is false, we will use regular phones (e.g. /ae/) as our basic phonetic unit.
# Otherwise, we will split them up to characters (e.g. /ae/ -> /a/, /e/).
phone_tokens=false
# When use_word_supervisions is true, we will add a language suffix to each word
# (e.g. "cat" -> "cat_English") and use these transcripts to train a word-level
# language model and the lang directory for model training.
# Otherwise, we will use phones themselves as "fake words"
# (e.g. text will be "k ae t" instead of "cat_English")
use_word_supervisions=false

# Acoustic model parameters
numLeavesTri1=1000
numGaussTri1=10000
numLeavesTri2=1000
numGaussTri2=20000
numLeavesTri3=6000
numGaussTri3=75000
numLeavesMLLT=6000
numGaussMLLT=75000
numLeavesSAT=6000
numGaussSAT=75000

langs_config="" # conf/experiments/all-ipa.conf
if [ $langs_config ]; then
  # shellcheck disable=SC1090
  source $langs_config
else
  # BABEL TRAIN:
  # Amharic - 307
  # Bengali - 103
  # Cantonese - 101
  # Javanese - 402
  # Vietnamese - 107
  # Zulu - 206
  # BABEL TEST:
  # Georgian - 404
  # Lao - 203
  babel_langs="307 103 101 402 107 206 404 203"
  babel_recog="${babel_langs}"
  gp_langs="Czech French Mandarin Spanish Thai"
  gp_recog="${gp_langs}"
  gp_path="/export/corpora5/GlobalPhone"
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
fi
###Globalphone####
#Czech       S0196
#French      S0197
#Spanish     S0203
#Mandarin    S0193
#Thai        S0321

. cmd.sh
. utils/parse_options.sh
. path.sh

local/install_shorten.sh

train_set=""
dev_set=""
for l in ${babel_langs}; do
  train_set="$l/data/train_${l} ${train_set}"
  dev_set="$l/data/dev_${l} ${dev_set}"
done
train_set_data=""
dev_set_data=""
for l in ${babel_langs}; do
  train_set_data="data/$l/data/train_${l} ${train_set_data}"
  dev_set_data="data/$l/data/dev_${l} ${dev_set_data}"
done
for l in ${gp_langs}; do
  train_set="GlobalPhone/gp_${l}_train ${train_set}"
  dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done
for l in ${gp_langs}; do
  train_set_data="data/GlobalPhone/gp_${l}_train ${train_set_data}"
  dev_set_data="data/GlobalPhone/gp_${l}_dev ${dev_set_data}"
done
train_set=${train_set%% }
dev_set=${dev_set%% }
train_set_data=${train_set_data%% }
dev_set_data=${dev_set_data%% }

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

echo "Training data directories: ${train_set[*]}"
echo "Dev data directories: ${dev_set[*]}"
echo "Eval data directories: ${recog_set[*]}"

full_train_set=train
full_dev_set=dev

function langname() {
  # Utility
  echo "$(basename "$1")"
}

phone_token_opt='--phones'
if [ $phone_tokens = true ]; then
  phone_token_opt='--phone-tokens'
fi

# This step will create the data directories for GlobalPhone and Babel languages.
# It's also going to use LanguageNet G2P models to convert text into phonetic transcripts.
# Depending on the settings, it will either transcribe into phones, e.g. ([m], [i:], [t]), or
# phonetic tokens, e.g. (/m/, /i/, /:/, /t/).
# The Kaldi "text" file will consist of these phonetic sequences, as we're trying to build
# a universal IPA recognizer.
# The lexicons are created separately for each split as an artifact from the ESPnet setup.
if (($stage <= 0)) && (($stop_stage > 0)); then
  echo "stage 0: Setting up individual languages"
  echo "babel_langs: $babel_langs"
  echo "gp_langs: $gp_langs"
  local/setup_languages.sh \
    --langs "${babel_langs}" \
    --recog "${babel_recog}" \
    --gp-langs "${gp_langs}" \
    --gp-recog "${gp_recog}" \
    --mboshi-train "${mboshi_train}" \
    --mboshi-recog "${mboshi_recog}" \
    --gp-romanized "${gp_romanized}" \
    --gp-path "${gp_path}" \
    --phone_token_opt "${phone_token_opt}" \
    --multilang true
  for x in ${train_set} ${dev_set} ${recog_set}; do
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done
fi

# Repair step if you changed your mind regarding word supervisions after running a few steps...

if $use_word_supervisions; then
  for data_dir in ${train_set}; do
    if [ -f data/$data_dir/text.bkp_suffix ]; then
      # replace IPA text with normal text (word having language suffix e.g. _Czech
      #cp data/$data_dir/text.bkp data/$data_dir/text
      cp data/$data_dir/text.bkp_suffix data/$data_dir/text
    fi
  done
else
  for data_dir in ${train_set}; do
    if [ -f data/$data_dir/text.bkp_suffix ]; then
      # replace IPA text with normal text (word having language suffix e.g. _Czech
      #cp data/$data_dir/text.bkp data/$data_dir/text
      cp data/$data_dir/text.ipa data/$data_dir/text
    fi
  done
fi

# Here we will combine the lexicons for train/dev/test splits into a single lexicon for each language.
if ((stage <= 1)) && ((stop_stage > 1)); then
  for data_dir in ${train_set}; do
    lang_name=$(langname $data_dir)
    mkdir -p data/local/$lang_name
    python3 local/combine_lexicons.py \
      data/$data_dir/lexicon_ipa.txt \
      data/${data_dir//train/dev}/lexicon_ipa.txt \
      data/${data_dir//train/eval}/lexicon_ipa.txt \
      >data/$data_dir/lexicon_ipa_all.txt
    python3 local/prepare_lexicon_dir.py $phone_token_opt data/$data_dir/lexicon_ipa_all.txt data/local/$lang_name
  done
fi

# We use the per-language lexicons to find the set of phones/phonetic tokens in every language and combine
# them again to obtain a multilingual "dummy" lexicon of the form:
# a a
# b b
# c c
# ...
# When that is ready, we train a multilingual phone-level language model (i.e. phonotactic model),
# that will be used to compile the decoding graph and to score each ASR system.
if ((stage <= 2)) && ((stop_stage > 2)); then
  local/prepare_ipa_lm.sh \
    --train-set "$train_set" \
    --phone_token_opt "$phone_token_opt" \
    --order "$phone_ngram_order"
  lexicon_list=$(find data/ipa_lm/train -name lexiconp.txt)
  mkdir -p data/local/dict_combined/local
  python3 local/combine_lexicons.py $lexicon_list >data/local/dict_combined/local/lexiconp.txt
  python3 local/prepare_lexicon_dir.py data/local/dict_combined/local/lexiconp.txt data/local/dict_combined
  utils/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_combined "<unk>" data/local/dict_combined data/lang_combined
  PHONE_LM=data/ipa_lm/train_all/srilm.o${phone_ngram_order}g.kn.gz
  utils/format_lm.sh data/lang_combined "$PHONE_LM" data/local/dict_combined/lexicon.txt data/lang_combined_test
fi

if (($stage <= 3)) && (($stop_stage > 3)); then
  #  We will generate a universal lexicon dir: data/local/lang_universal and
  #                      a universal lang dir: data/lang_universal;
  #  data/lang_universal/words.txt come from multiple languages and each with a language suffix like _101.
  #  Pronunciations in data/lang_universal/phones/align_lexicon.txt use IPA phone symbols, same as in monolingual recipe
  mkdir -p data/local/lang_universal
  for data_dir in ${train_set}; do
    lang_name="$(langname $data_dir)"
    cp data/$data_dir/lexicon_ipa_suffix.txt data/local/lang_universal/lexicon_ipa_suffix_${lang_name}.txt
  done
  # Create a language-universal lexicon; each word has a language-suffix like "word_English word_Czech";
  # Because of that we can just concatenate and sort the lexicons.
  cat data/local/lang_universal/lexicon_ipa_suffix*.txt |
    sort \
      >data/local/lang_universal/lexicon_ipa_suffix_universal.txt
  # Create a regular Kaldi dict dir using the combined lexicon.
  python3 local/prepare_lexicon_dir.py \
    $phone_token_opt \
    data/local/lang_universal/lexicon_ipa_suffix_universal.txt \
    data/local/lang_universal
  # Create a regular Kaldi lang dir using the combined lexicon.
  utils/prepare_lang.sh \
    --position-dependent-phones false \
    --share-silence-phones true \
    data/local/lang_universal '<unk>' data/local/tmp.lang_universal data/lang_universal
  # Train the LM and evaluate on the dev set transcripts
  local/prepare_word_lm.sh \
    --train-set "$train_set" \
    --order "$word_ngram_order"
  WORD_LM=data/word_lm/train_all/srilm.o${word_ngram_order}g.kn.gz
  utils/format_lm.sh data/lang_universal "$WORD_LM" data/local/lang_universal/lexicon_ipa_suffix_universal.txt data/lang_universal_test
fi

if (($stage <= 4)) && (($stop_stage > 4)); then
  # Feature extraction
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/make_mfcc.sh \
        --cmd "$train_cmd" \
        --nj $extract_feat_nj \
        --write_utt2num_frames true \
        "data/$data_dir" \
        "exp/make_mfcc/$data_dir" \
        mfcc
      utils/fix_data_dir.sh data/$data_dir
      steps/compute_cmvn_stats.sh data/$data_dir exp/make_mfcc/$lang_name mfcc/$lang_name
    ) &
    sleep 2
  done
  wait
fi

if (($stage <= 5)) && (($stop_stage > 5)); then
  echo "combine data dirs to a universal data dir in data/universal"
  echo "train_set_data: $train_set_data"
  utils/combine_data.sh data/universal/train $train_set_data
  utils/validate_data_dir.sh data/universal/train || exit 1
  echo "$train_set" >data/universal/train/original_data_dirs.txt
fi

if (($stage <= 6)) && (($stop_stage > 6)); then
  # Prepare data dir subsets for monolingual training
  numutt=$(cat data/universal/train/feats.scp | wc -l)
  if [ $numutt -gt 50000 ]; then
    utils/subset_data_dir.sh data/universal/train 50000 data/subsets/50k/universal/train
  else
    mkdir -p "$(dirname data/subsets/50k/universal/train)"
    ln -s "$(pwd)/data/universal/train" "data/subsets/50k/universal/train"
  fi
  if [ $numutt -gt 100000 ]; then
    utils/subset_data_dir.sh data/universal/train 100000 data/subsets/100k/universal/train
  else
    mkdir -p "$(dirname data/subsets/100k/universal/train)"
    ln -s "$(pwd)/data/universal/train" "data/subsets/100k/universal/train"
  fi
  if [ $numutt -gt 200000 ]; then
    utils/subset_data_dir.sh data/universal/train 200000 data/subsets/200k/universal/train
  else
    mkdir -p "$(dirname data/subsets/200k/universal/train)"
    ln -s "$(pwd)/data/universal/train" "data/subsets/200k/universal/train"
  fi
fi

lang=data/lang_combined_test
if $use_word_supervisions; then
  lang=data/lang_universal_test
fi

data_dir=universal/train
if (($stage <= 7)) && (($stop_stage > 7)); then
  # Mono training
  expdir=exp/gmm/mono
  steps/train_mono.sh \
    --nj $early_train_nj --cmd "$train_cmd" \
    data/subsets/50k/$data_dir \
    $lang $expdir
fi

if (($stage <= 8)) && (($stop_stage > 8)); then
  # Tri1 training
  steps/align_si.sh \
    --nj $early_train_nj --cmd "$train_cmd" \
    data/subsets/100k/$data_dir \
    $lang \
    exp/gmm/mono \
    exp/gmm/mono_ali_100k

  steps/train_deltas.sh \
    --cmd "$train_cmd" \
    $numLeavesTri1 \
    $numGaussTri1 \
    data/subsets/100k/$data_dir \
    $lang \
    exp/gmm/mono_ali_100k \
    exp/gmm/tri1
fi

if (($stage <= 9)) && (($stop_stage > 9)); then
  # Tri2 training
  steps/align_si.sh \
    --nj $early_train_nj --cmd "$train_cmd" \
    data/subsets/200k/$data_dir \
    $lang \
    exp/gmm/tri1 \
    exp/gmm/tri1_ali_200k

  steps/train_deltas.sh \
    --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/subsets/200k/$data_dir \
    $lang \
    exp/gmm/tri1_ali_200k \
    exp/gmm/tri2
fi

if (($stage <= 10)) && (($stop_stage > 10)); then
  # Tri3 training
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    exp/gmm/tri2 \
    exp/gmm/tri2_ali

  steps/train_deltas.sh \
    --cmd "$train_cmd" $numLeavesTri3 $numGaussTri3 \
    data/$data_dir \
    $lang \
    exp/gmm/tri2_ali \
    exp/gmm/tri3
fi

if (($stage <= 11)) && (($stop_stage > 11)); then
  # Tri4 training
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    exp/gmm/tri3 \
    exp/gmm/tri3_ali

  steps/train_lda_mllt.sh \
    --cmd "$train_cmd" \
    $numLeavesMLLT \
    $numGaussMLLT \
    data/$data_dir \
    $lang \
    exp/gmm/tri3_ali \
    exp/gmm/tri4
fi

if (($stage <= 12)) && (($stop_stage > 12)); then
  # Tri5 training
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    exp/gmm/tri4 \
    exp/gmm/tri4_ali

  steps/train_sat.sh \
    --cmd "$train_cmd" \
    $numLeavesSAT \
    $numGaussSAT \
    data/$data_dir \
    $lang \
    exp/gmm/tri4_ali \
    exp/gmm/tri5
fi

if (($stage <= 13)) && (($stop_stage > 13)); then
  # Tri5 alignments
  steps/align_fmllr.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    exp/gmm/tri5 \
    exp/gmm/tri5_ali
fi

# Uncomment this if you intend to train Chain TDNNF AM in next steps
# bash local/chain_multilang/tuning/run_tdnn_1g.sh --langdir $lang
