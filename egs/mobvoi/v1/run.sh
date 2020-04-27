#!/bin/bash
# Copyright 2018-2020  Daniel Povey
#           2018-2020  Yiming Wang

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
    mv $dir/text.tmp $dir/text || exit 1
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
  id_sil=`cat data/lang/words.txt | grep "<sil>" | awk '{print $2}'`
  id_freetext=`cat data/lang/words.txt | grep "FREETEXT" | awk '{print $2}'`
  export LC_ALL=en_US.UTF-8
  id_word=`cat data/lang/words.txt | grep "嗨小问" | awk '{print $2}'`
  export LC_ALL=C
  mkdir -p data/lang/lm
  cat <<EOF > data/lang/lm/fst.txt
0 1 $id_sil $id_sil
0 4 $id_sil $id_sil 7.0
1 4 $id_freetext $id_freetext 0.0
4 0 $id_sil $id_sil
1 2 $id_word $id_word 1.1
2 0 $id_sil $id_sil
0
EOF
  fstcompile data/lang/lm/fst.txt data/lang/G.fst
  set +e
  fstisstochastic data/lang/G.fst
  set -e
  utils/validate_lang.pl data/lang
fi

if [ $stage -le 7 ]; then
  echo "$0: subsegmenting for the training data..."
  srcdir=data/train
  utils/data/convert_data_dir_to_whole.sh $srcdir ${srcdir}_whole

  utils/data/get_segments_for_data.sh $srcdir > ${srcdir}_whole/segments
  utils/filter_scp.pl <(awk '{if ($2 == "FREETEXT") print $1}' ${srcdir}_whole/text) \
    ${srcdir}_whole/segments >${srcdir}_whole/neg_segments
  utils/filter_scp.pl --exclude ${srcdir}_whole/neg_segments ${srcdir}_whole/segments \
    >${srcdir}_whole/pos_segments
  utils/filter_scp.pl ${srcdir}_whole/pos_segments ${srcdir}_whole/utt2dur >${srcdir}_whole/pos_utt2dur
  local/get_random_subsegments.py --overlap-duration=0.3 --max-remaining-duration=0.3 \
    ${srcdir}_whole/neg_segments ${srcdir}_whole/pos_utt2dur | \
    cat ${srcdir}_whole/pos_segments - | sort >${srcdir}_whole/sub_segments
  utils/data/subsegment_data_dir.sh ${srcdir}_whole \
    ${srcdir}_whole/sub_segments data/train_segmented
  awk '{print $1,$2}' ${srcdir}_whole/sub_segments | \
    utils/apply_map.pl -f 2 ${srcdir}_whole/text >data/train_segmented/text
  utils/data/extract_wav_segments_data_dir.sh --nj 50 --cmd "$train_cmd" \
    data/train_segmented data/train_shorter
  steps/compute_cmvn_stats.sh data/train_shorter
  utils/fix_data_dir.sh data/train_shorter
  utils/validate_data_dir.sh data/train_shorter
fi

# In this section, we augment the training data with reverberation,
# noise, music, and babble, and combined it with the clean data.
if [ $stage -le 8 ]; then
  utils/data/get_utt2dur.sh data/train_shorter
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
    data/train_shorter data/train_shorter_reverb
  cat data/train_shorter/utt2dur | awk -v name=rev1 '{print name"-"$0}' >data/train_shorter_reverb/utt2dur

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    cp data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  export LC_ALL=en_US.UTF-8
  steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id true --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train_shorter data/train_shorter_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-prefix "music" --modify-spk-id true --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train_shorter data/train_shorter_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spk-id true --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train_shorter data/train_shorter_babble
  export LC_ALL=C
fi

if [ $stage -le 9 ]; then
  # Now make MFCC features
  for name in reverb noise music babble; do
    steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" \
      data/train_shorter_${name} || exit 1;
    steps/compute_cmvn_stats.sh data/train_shorter_${name}
    utils/fix_data_dir.sh data/train_shorter_${name}
    utils/validate_data_dir.sh data/train_shorter_${name}
  done
fi

# monophone training
if [ $stage -le 10 ]; then
  steps/train_mono.sh --nj 50 --cmd "$train_cmd" \
    data/train_shorter data/lang exp/mono
  (
    utils/mkgraph.sh data/lang \
      exp/mono exp/mono/graph
  )&

  steps/align_si.sh --nj 50 --cmd "$train_cmd" \
    data/train_shorter data/lang exp/mono exp/mono_ali_train_shorter
fi

if [ $stage -le 11 ]; then
  echo "$0: preparing for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/train_shorter data/train_shorter_sp
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 data/train_shorter_sp || exit 1;
  steps/compute_cmvn_stats.sh data/train_shorter_sp || exit 1;
  utils/fix_data_dir.sh data/train_shorter_sp
fi

if [ $stage -le 12 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
    data/train_shorter_sp data/lang exp/mono exp/mono_ali_train_shorter_sp || exit 1
fi

if [ $stage -le 13 ]; then
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/train_shorter_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/mobvoi-$(dte +'%m_%d_%H_%M')/v1/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in train_shorter_sp dev eval; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/train_shorter_sp_hires || exit 1;

  for datadir in train_shorter_sp dev eval; do
    steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi

combined_train_set=train_shorter_sp_combined
aug_affix="reverb noise music babble"
if [ $stage -le 14 ]; then
  for name in $aug_affix; do
    echo "$0: creating high-resolution MFCC features for train_shorter_${name}"
    mfccdir=data/train_shorter_${name}_hires/data
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/mobvoi-$(date +'%m_%d_%H_%M')/v1/$mfccdir/storage $mfccdir/storage
    fi
    utils/copy_data_dir.sh data/train_shorter_${name} data/train_shorter_${name}_hires
    steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_shorter_${name}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/train_shorter_${name}_hires || exit 1;
    utils/fix_data_dir.sh data/train_shorter_${name}_hires || exit 1;
  done
  eval utils/combine_data.sh data/${combined_train_set}_hires data/train_shorter_sp_hires \
    data/train_shorter_{$(echo $aug_affix | sed 's/ /,/g')}_hires
fi


if [ $stage -le 15 ]; then
  local/chain/run_tdnn.sh --train-set train_shorter --combined-train-set ${combined_train_set}
fi

exit 0

