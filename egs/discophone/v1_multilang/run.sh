#!/bin/bash

set -eou pipefail

stage=0
stop_stage=1
train_nj=24
extract_feat_nj=8
train_mono_nj=1
train_tri2_nj=1

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
#  babel_langs="307 103 101 402 107 206 404 203"
#  babel_recog="${babel_langs}"
  #babel_langs=""
  #babel_recog=""
  gp_langs=""
  gp_recog=""
  gp_path="/export/corpora5/GlobalPhone"
  #gp_langs="Czech French Mandarin Spanish Thai"
  #gp_recog="${gp_langs}"
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
  ipa_transcript=true
fi
###Globalphone####
#Czech       S0196
#French      S0197
#Spanish     S0203
#Mandarin    S0193
#Thai        S0321

babel_langs="307 103 101 402 107 206 404 203"
babel_recog="${babel_langs}"
. cmd.sh
. utils/parse_options.sh
. path.sh

local/install_shorten.sh

# TODO: copy data dir creation from ESPnet discophone

train_set=""
dev_set=""
for l in ${babel_langs}; do
  train_set="$l/data/train_${l} ${train_set}"
  dev_set="$l/data/dev_${l} ${dev_set}"
done
train_set_data=""
dev_set_data=""
for l in ${babel_langs} ;do
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

if (($stage <= 0)) && (($stop_stage > 0 ))  ; then
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
    --ipa-transcript "${ipa_transcript}" \
    --gp-path "${gp_path}"
  for x in ${train_set} ${dev_set} ${recog_set}; do
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done

  for data_dir in ${train_set}; do
    if [ -f data/$data_dir/text.bkp_suffix ]; then
      # replace IPA text with normal text (word having language suffix e.g. _Czech
      #cp data/$data_dir/text.bkp data/$data_dir/text
      cp data/$data_dir/text.bkp_suffix data/$data_dir/text
    fi
  done
fi

if (($stage <= 4)) && (($stop_stage > 4 )) ; then
#  We will generate a universal lexicon dir: data/local/lang_universal and 
#                      a universal lang dir: data/lang_universal; 
#  data/lang_universal/words.txt come from multiple languages and each with a language suffix like _101. Pronunciations in data/lang_universal/phones/align_lexicon.txt use IPA phone symbols, same as in monolingual recipe 
  mkdir -p data/local/lang_universal
  for data_dir in ${train_set}; do
    lang_name="$(langname $data_dir)"
    cp data/$data_dir/lexicon_ipa_suffix.txt data/local/lang_universal/lexicon_ipa_suffix_${lang_name}.txt
  done
  cat data/local/lang_universal/lexicon_ipa_suffix*.txt > data/local/lang_universal/lexicon_ipa_suffix_universal.txt
  python local/prepare_lexicon_dir.py --phone-tokens data/local/lang_universal/lexicon_ipa_suffix_universal.txt data/local/lang_universal
  utils/prepare_lang.sh \
    --share-silence-phones true \
    data/local/lang_universal '<unk>' data/local/tmp.lang_universal data/lang_universal
fi

if (($stage <= 5)) && (($stop_stage > 5 )) ; then
  # Feature extraction
#  for data_dir in ${train_set}; do
#    #(
#      lang_name=$(langname $data_dir)
#      steps/make_mfcc.sh \
#        --cmd "$train_cmd" \
#        --nj $extract_feat_nj \
#        --write_utt2num_frames true \
#        data/$data_dir \
#        exp/make_mfcc/$data_dir \
#        mfcc
#      utils/fix_data_dir.sh data/$data_dir
#      steps/compute_cmvn_stats.sh data/$data_dir exp/make_mfcc/$lang_name mfcc/$lang_name
#   # ) &
#   # sleep 2
#  done
#  wait
  echo "combine data dirs to a universal data dir in data/universal"
  echo "train_set_data: $train_set_data"
  utils/combine_data.sh data/universal/train $train_set_data 
  utils/validate_data_dir.sh data/universal/train || exit 1;
  echo "$train_set" > data/universal/train/original_data_dirs.txt   

#  We don't merge dev or eval data from multiple languages here
#  echo "dev_set_data: $dev_set_data" 
#  utils/combine_data.sh data/universal/dev $dev_set_data
#  echo "$dev_set" > data/universal/dev/original_data_dirs.txt 
fi

if (($stage <= 6)) && (($stop_stage > 6 )) ; then
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




#######Replace below with 1-thread
# From now on,  all data/lang/$lang_name is replaced by data/lang_universal
data_dir=universal/train 
if (($stage <= 7))  && (($stop_stage > 7 )) ; then
  # Mono training
      expdir=exp/gmm/mono
      steps/train_mono.sh \
        --nj $train_mono_nj --cmd "$train_cmd" \
        data/subsets/50k/$data_dir \
        data/lang_universal $expdir
fi

if (($stage <= 8))  && (($stop_stage > 8 )) ; then
  # Tri1 training
      steps/align_si.sh \
        --nj $train_mono_nj --cmd "$train_cmd" \
        data/subsets/100k/$data_dir \
        data/lang_universal \
        exp/gmm/mono \
        exp/gmm/mono_ali_100k

      steps/train_deltas.sh \
        --cmd "$train_cmd" \
        $numLeavesTri1 \
        $numGaussTri1 \
        data/subsets/100k/$data_dir \
        data/lang_universal \
        exp/gmm/mono_ali_100k \
        exp/gmm/tri1
fi

if (($stage <= 9)) && (($stop_stage > 9 )) ; then
  # Tri2 training
      steps/align_si.sh \
        --nj $train_tri2_nj --cmd "$train_cmd" \
        data/subsets/200k/$data_dir \
        data/lang_universal \
        exp/gmm/tri1 \
        exp/gmm/tri1_ali_200k

      steps/train_deltas.sh \
        --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
        data/subsets/200k/$data_dir \
        data/lang_universal \
        exp/gmm/tri1_ali_200k \
        exp/gmm/tri2

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/subsets/200k/$data_dir \
        data/lang_universal \
        data/local/lang_universal \
        exp/gmm/tri2 \
        data/local/dictp/lang_universal/tri2 \
        data/local/langp/lang_universal/tri2 \
        data/langp/lang_universal/tri2
  wait
fi

if (($stage <= 10)) && (($stop_stage > 10 )) ; then
  # Tri3 training
      steps/align_si.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/lang_universal/tri2 \
        exp/gmm/tri2 \
        exp/gmm/tri2_ali

      steps/train_deltas.sh \
        --cmd "$train_cmd" $numLeavesTri3 $numGaussTri3 \
        data/$data_dir \
        data/langp/lang_universal/tri2 \
        exp/gmm/tri2_ali \
        exp/gmm/tri3

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang_universal \
        data/local/lang_universal \
        exp/gmm/tri3 \
        data/local/dictp/lang_universal/tri3 \
        data/local/langp/lang_universal/tri3 \
        data/langp/lang_universal/tri3
fi

if (($stage <= 11)) && (($stop_stage > 11 )) ; then
  # Tri4 training
      steps/align_si.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/lang_universal/tri3 \
        exp/gmm/tri3 \
        exp/gmm/tri3_ali

      steps/train_lda_mllt.sh \
        --cmd "$train_cmd" \
        $numLeavesMLLT \
        $numGaussMLLT \
        data/$data_dir \
        data/langp/lang_universal/tri3 \
        exp/gmm/tri3_ali \
        exp/gmm/tri4

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang_universal \
        data/local/lang_universal \
        exp/gmm/tri4 \
        data/local/dictp/lang_universal/tri4 \
        data/local/langp/lang_universal/tri4 \
        data/langp/lang_universal/tri4
fi

if (($stage <= 12)) && (($stop_stage > 12 )) ; then
  # Tri5 training
      steps/align_si.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/lang_universal/tri4 \
        exp/gmm/tri4 \
        exp/gmm/tri4_ali

      steps/train_sat.sh \
        --cmd "$train_cmd" \
        $numLeavesSAT \
        $numGaussSAT \
        data/$data_dir \
        data/langp/lang_universal/tri4 \
        exp/gmm/tri4_ali \
        exp/gmm/tri5

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang_universal \
        data/local/lang_universal \
        exp/gmm/tri5 \
        data/local/dictp/lang_universal/tri5 \
        data/local/langp/lang_universal/tri5 \
        data/langp/lang_universal/tri5
  wait
fi

if (($stage <= 13)) && (($stop_stage > 13 )) ; then
  # Tri5 alignments
      steps/align_fmllr.sh \
        --nj $train_nj --cmd "$train_cmd" \
        data/$data_dir \
        data/langp/lang_universal/tri5 \
        exp/gmm/tri5 \
        exp/gmm/tri5_ali

      local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
        data/$data_dir \
        data/lang_universal \
        data/local/lang_universal \
        exp/gmm/tri5_ali \
        data/local/dictp/lang_universal/tri5_ali \
        data/local/langp/lang_universal/tri5_ali \
        data/langp/lang_universal/tri5_ali
fi


