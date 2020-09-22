#!/bin/bash

set -eou pipefail

stage=0
train_nj=24
phone_tokens=false

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
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
fi

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
for l in ${gp_langs}; do
  train_set="GlobalPhone/gp_${l}_train ${train_set}"
  dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done
train_set=${train_set%% }
dev_set=${dev_set%% }

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

phone_token_opt=
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
if ((stage <= 0)); then
  echo "stage 0: Setting up individual languages"
  local/setup_languages.sh \
    --langs "${babel_langs}" \
    --recog "${babel_recog}" \
    --gp-langs "${gp_langs}" \
    --gp-recog "${gp_recog}" \
    --mboshi-train "${mboshi_train}" \
    --mboshi-recog "${mboshi_recog}" \
    --gp-romanized "${gp_romanized}" \
    --phone_token_opt "${phone_token_opt}"
  for x in ${train_set} ${dev_set} ${recog_set}; do
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done
fi

# Here we will combine the lexicons for train/dev/test splits into a single lexicon for each language.
if ((stage <= 2)); then
  for data_dir in ${train_set}; do
    lang_name=$(langname $data_dir)
    mkdir -p data/local/$lang_name
    python3 local/combine_lexicons.py \
      data/$data_dir/lexicon_ipa.txt \
      data/${data_dir//train/dev}/lexicon_ipa.txt \
      data/${data_dir//train/eval}/lexicon_ipa.txt \
      >data/$data_dir/lexicon_ipa_all.txt
    python3 local/prepare_lexicon_dir.py $phone_token_opt data/$data_dir/lexicon_ipa_all.txt data/local/$lang_name
    lang_name="$(langname $data_dir)"
    utils/prepare_lang.sh \
      --share-silence-phones true \
      data/local/$lang_name '<unk>' data/local/tmp.lang/$lang_name data/lang/$lang_name
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
if ((stage <= 3)); then
  local/prepare_ipa_lm.sh --train-set "$train_set" --dev-set "$dev_set" --phone_token_opt "$phone_token_opt"
  lexicon_list=$(find data/ipa_lm/train -name lexiconp.txt)
  mkdir -p data/local/dict_combined/local
  python3 local/combine_lexicons.py $lexicon_list >data/local/dict_combined/local/lexiconp.txt
  python3 local/prepare_lexicon_dir.py data/local/dict_combined/local/lexiconp.txt data/local/dict_combined
  utils/prepare_lang.sh \
    --position-dependent-phones false \
    --share-silence-phones true \
    data/local/dict_combined \
    "<unk>" data/local/dict_combined data/lang_combined
  LM=data/ipa_lm/train_all/srilm.o3g.kn.gz
  utils/format_lm.sh data/lang_combined "$LM" data/local/dict_combined/lexicon.txt data/lang_combined_test
fi

# MFCC extraction for GMM training
if ((stage <= 5)); then
  # Feature extraction
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/make_mfcc.sh \
        --cmd "$train_cmd" \
        --nj 16 \
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

if ((stage <= 6)); then
  # Prepare data dir subsets for monolingual training
  for data_dir in ${train_set}; do
    numutt=$(cat data/$data_dir/feats.scp | wc -l)
    if [ $numutt -gt 5000 ]; then
      utils/subset_data_dir.sh data/$data_dir 5000 data/subsets/5k/$data_dir
    else
      mkdir -p "$(dirname data/subsets/5k/$data_dir)"
      if [ ! -L "data/subsets/5k/$data_dir" ]; then
        ln -s "$(pwd)/data/$data_dir" "data/subsets/5k/$data_dir"
      fi
    fi
    if [ $numutt -gt 10000 ]; then
      utils/subset_data_dir.sh data/$data_dir 10000 data/subsets/10k/$data_dir
    else
      mkdir -p "$(dirname data/subsets/10k/$data_dir)"
      if [ ! -L "data/subsets/10k/$data_dir" ]; then
        ln -s "$(pwd)/data/$data_dir" "data/subsets/10k/$data_dir"
      fi
    fi
    if [ $numutt -gt 20000 ]; then
      utils/subset_data_dir.sh data/$data_dir 20000 data/subsets/20k/$data_dir
    else
      mkdir -p "$(dirname data/subsets/20k/$data_dir)"
      if [ ! -L "data/subsets/20k/$data_dir" ]; then
        ln -s "$(pwd)/data/$data_dir" "data/subsets/20k/$data_dir"
      fi
    fi
  done
fi

if ((stage <= 7)); then
  # Mono training
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/mono
      steps/train_mono.sh \
        --nj 8 --cmd "$train_cmd" \
        data/subsets/5k/$data_dir \
        data/lang/$lang_name $expdir
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 8)); then
  # Tri1 training
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/align_si.sh \
        --nj 12 --cmd "$train_cmd" \
        data/subsets/10k/$data_dir \
        data/lang/$lang_name \
        exp/gmm/$lang_name/mono \
        exp/gmm/$lang_name/mono_ali_10k

      steps/train_deltas.sh \
        --cmd "$train_cmd" \
        $numLeavesTri1 \
        $numGaussTri1 \
        data/subsets/10k/$data_dir \
        data/lang/$lang_name \
        exp/gmm/$lang_name/mono_ali_10k \
        exp/gmm/$lang_name/tri1
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 9)); then
  # Tri2 training
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/align_si.sh \
        --nj 24 --cmd "$train_cmd" \
        data/subsets/20k/$data_dir \
        data/lang/$lang_name \
        exp/gmm/$lang_name/tri1 \
        exp/gmm/$lang_name/tri1_ali_20k

      steps/train_deltas.sh \
        --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
        data/subsets/20k/$data_dir \
        data/lang/$lang_name \
        exp/gmm/$lang_name/tri1_ali_20k \
        exp/gmm/$lang_name/tri2

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/subsets/20k/$data_dir \
        data/lang/$lang_name \
        data/local/$lang_name \
        exp/gmm/$lang_name/tri2 \
        data/local/dictp/$lang_name/tri2 \
        data/local/langp/$lang_name/tri2 \
        data/langp/$lang_name/tri2
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 10)); then
  # Tri3 training
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/align_si.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/$lang_name/tri2 \
        exp/gmm/$lang_name/tri2 \
        exp/gmm/$lang_name/tri2_ali

      steps/train_deltas.sh \
        --cmd "$train_cmd" $numLeavesTri3 $numGaussTri3 \
        data/$data_dir \
        data/langp/$lang_name/tri2 \
        exp/gmm/$lang_name/tri2_ali \
        exp/gmm/$lang_name/tri3

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang/$lang_name \
        data/local/$lang_name \
        exp/gmm/$lang_name/tri3 \
        data/local/dictp/$lang_name/tri3 \
        data/local/langp/$lang_name/tri3 \
        data/langp/$lang_name/tri3
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 11)); then
  # Tri4 training
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/align_si.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/$lang_name/tri3 \
        exp/gmm/$lang_name/tri3 \
        exp/gmm/$lang_name/tri3_ali

      steps/train_lda_mllt.sh \
        --cmd "$train_cmd" \
        $numLeavesMLLT \
        $numGaussMLLT \
        data/$data_dir \
        data/langp/$lang_name/tri3 \
        exp/gmm/$lang_name/tri3_ali \
        exp/gmm/$lang_name/tri4

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang/$lang_name \
        data/local/$lang_name \
        exp/gmm/$lang_name/tri4 \
        data/local/dictp/$lang_name/tri4 \
        data/local/langp/$lang_name/tri4 \
        data/langp/$lang_name/tri4
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 12)); then
  # Tri5 training
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/align_si.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/$lang_name/tri4 \
        exp/gmm/$lang_name/tri4 \
        exp/gmm/$lang_name/tri4_ali

      steps/train_sat.sh \
        --cmd "$train_cmd" \
        $numLeavesSAT \
        $numGaussSAT \
        data/$data_dir \
        data/langp/$lang_name/tri4 \
        exp/gmm/$lang_name/tri4_ali \
        exp/gmm/$lang_name/tri5

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang/$lang_name \
        data/local/$lang_name \
        exp/gmm/$lang_name/tri5 \
        data/local/dictp/$lang_name/tri5 \
        data/local/langp/$lang_name/tri5 \
        data/langp/$lang_name/tri5
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 13)); then
  # Tri5 alignments
  for data_dir in ${train_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/align_fmllr.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/$lang_name/tri5 \
        exp/gmm/$lang_name/tri5 \
        exp/gmm/$lang_name/tri5_ali

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang/$lang_name \
        data/local/$lang_name \
        exp/gmm/$lang_name/tri5_ali \
        data/local/dictp/$lang_name/tri5_ali \
        data/local/langp/$lang_name/tri5_ali \
        data/langp/$lang_name/tri5_ali
    ) &
    sleep 2
  done
  wait
fi
