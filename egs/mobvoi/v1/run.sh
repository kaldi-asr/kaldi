#!/bin/bash

stage=0


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  local/mobvoi_data_download.sh
  echo "$0: Extracted all datasets into data/download/"
fi

if [ $stage -le 1 ]; then
  echo "$0: Splitting datasets..."
  local/split_datasets.sh
  echo "$0: text and utt2spk have been generated in data/{train|dev|eval}."
fi
    
if [ $stage -le 2 ]; then
  echo "$0: Preparing wav.scp..."
  local/prepare_wav.py data
  echo "wav.scp has been generated in data/{train|dev|eval}."
fi

if [ $stage -le 3 ]; then
  echo "$0: Extracting MFCC..."
  for folder in train dev eval; do
    dir=data/$folder
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $dir
    steps/compute_cmvn_stats.sh $dir
    utils/fix_data_dir.sh $dir
    utils/data/get_utt2dur.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: Post processing transcripts..."
  for folder in train dev eval; do
    dir=data/$folder
    cat $dir/text | awk '{if ($2=="嗨小问" || $2=="嗨小问嗨小问") {print $1,"嗨小问";} else {print $1,"FREETEXT"}}' > $dir/text.tmp || exit 1
    cat $dir/text.tmp > $dir/text || exit 1
    rm -f $dir/text.tmp 2>/dev/null || true
  done
fi

if [ $stage -le 5 ]; then
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 1 --num-nonsil-states 4 --sil-prob 0.5 \
    --position-dependent-phones false \
    data/local/dict "<sil>" data/lang/temp data/lang
fi

if [ $stage -le 6 ]; then
  id_freetext=`cat data/lang/words.txt | grep "FREETEXT" | awk '{print $2}'`
  id_word=`cat data/lang/words.txt | grep "嗨小问" | awk '{print $2}'`
  mkdir -p data/lang/lm
  cat <<EOF > data/lang/lm/fst.txt
0 1 $id_freetext $id_freetext
0 2 $id_word $id_word
1 0.0
2 1.1
EOF
  fstcompile data/lang/lm/fst.txt data/lang/G.fst
  set +e
  fstisstochastic data/lang/G.fst
  set -e
  utils/validate_lang.pl data/lang
  exit 0
fi

# In this section, we augment the training data with reverberation,
# noise, music, and babble, and combined it with the clean data.
if [ $stage -le 7 ]; then
  cp data/train/utt2dur data/train/reco2dur
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  [ ! -f rirs_noises.zip ] && wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  [ ! -d "RIRS_NOISES" ] && unzip rirs_noises.zip

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "rev" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cat data/train/utt2dur | awk -v name=rev1 '{print name"_"$0}' >data/train_reverb/utt2dur

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    cp data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir_for_asr.py --utt-prefix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  cat data/train/utt2dur | awk -v name=noise '{print name"_"$0}' >data/train_noise/utt2dur
  #awk '{a=$1; sub(/-noise$/, "", $1); print a, $1}' data/train_noise/wav.scp > data/train_noise/utt2uniq
  # Augment with musan_music
  steps/data/augment_data_dir_for_asr.py --utt-prefix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  cat data/train/utt2dur | awk -v name=music '{print name"_"$0}' >data/train_music/utt2dur
  #awk '{a=$1; sub(/-music$/, "", $1); print a, $1}' data/train_noise/wav.scp > data/train_music/utt2uniq
  # Augment with musan_speech
  steps/data/augment_data_dir_for_asr.py --utt-prefix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble
  cat data/train/utt2dur | awk -v name=babble '{print name"_"$0}' >data/train_babble/utt2dur
fi

if [ $stage -le 8 ]; then
  # Now make MFCC features

  for name in reverb noise music babble; do
    steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" \
      data/train_${name} || exit 1;
    steps/compute_cmvn_stats.sh data/train_${name}
    utils/fix_data_dir.sh data/train_${name}
    utils/validate_data_dir.sh data/train_${name}
  done
fi

# monophone training
if [ $stage -le 9 ]; then
  steps/train_mono.sh --nj 50 --cmd "$train_cmd" \
    data/train data/lang exp/mono
  (
    utils/mkgraph.sh data/lang \
      exp/mono exp/mono/graph
    for test in dev eval; do
      steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/mono/graph \
        data/$test exp/mono/decode_$test
    done
  )&

  steps/align_si.sh --nj 50 --cmd "$train_cmd" \
    data/train_combined data/lang exp/mono exp/mono_ali
fi

exit 0
# train an LDA+MLLT system.
if [ $stage -le 10 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2000 30000 \
    data/train data/lang exp/mono_ali exp/mono2

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang exp/mono2 exp/mono2/graph
    for test in dev eval; do
      steps/decode.sh --nj 50 --cmd "$decode_cmd" exp/mono2/graph \
        data/$test exp/mono2/decode_$test
    done
  )&

  steps/align_si.sh  --nj 50 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/mono2 exp/mono2_ali
fi

trainset=train
if [ -f data/${trainset}_spe2e_hires/feats.scp ]; then
  echo "$0: It seems that features for the perturbed training data already exist."
  echo "If you want to extract them anyway, remove them first and run this"
  echo "stage again. Skipping this stage..."
else
  if [ $stage -le 11 ]; then
    echo "$0: perturbing the training data to allowed lengths..."
    utils/data/get_utt2dur.sh data/${trainset}  # necessary for the next command

    # 12 in the following command means the allowed lengths are spaced
    # by 12% change in length.
    export LC_ALL=en_US.UTF-8
    utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset} \
                                                   data/${trainset}_spe2e_hires
    export LC_ALL=C
    cat data/${trainset}_spe2e_hires/utt2dur | \
      awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_hires/utt2uniq
    utils/fix_data_dir.sh data/${trainset}_spe2e_hires
  fi

  if [ $stage -le 12 ]; then
    echo "$0: extracting MFCC features for the training data..."
    steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
                       --cmd "$train_cmd" data/${trainset}_spe2e_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e_hires || exit 1;
    utils/fix_data_dir.sh data/${trainset}_spe2e_hires
    utils/validate_data_dir.sh data/${trainset}_spe2e_hires
  fi
fi

if [ $stage -le 13 ]; then
  echo "$0: extracting MFCC features for the dev/eval data..."
  for datadir in train dev eval; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  for datadir in train dev eval; do
    steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires
  done
  #exit 0
fi

if [ $stage -le 14 ]; then
  echo "$0: Calling the chain recipe..."
  local/chain/run_tdnn_1a.sh
fi


