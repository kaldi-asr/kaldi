#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares Fisher data for training speech activity detection,
# music detection, and overlapped speech detection systems.

. path.sh
. cmd.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  echo "This script is to serve as an example recipe."
  echo "Edit the script to change variables if needed."
  exit 1
fi

dir=exp/unsad/make_unsad_fisher_train_100k  # Work dir
subset=60   # Number of recordings to keep before speed perturbation and corruption
utt_subset=75000  # Number of utterances to keep after speed perturbation for adding overlapped-speech

# All the paths below can be modified to any absolute path.

# The original data directory which will be converted to a whole (recording-level) directory.
train_data_dir=data/fisher_train_100k   

model_dir=exp/tri3a   # Model directory used for decoding
sat_model_dir=exp/tri4a   # Model directory used for getting alignments
lang=data/lang  # Language directory
lang_test=data/lang_test  # Language directory used to build graph

# Hard code the mapping from phones to SAD labels
# 0 for silence, 1 for speech, 2 for noise, 3 for unk
cat <<EOF > $dir/fisher_sad.map
sil 0
sil_B 0
sil_E 0
sil_I 0
sil_S 0
laughter 2
laughter_B 2
laughter_E 2
laughter_I 2
laughter_S 2
noise 2
noise_B 2
noise_E 2
noise_I 2
noise_S 2
oov 3
oov_B 3
oov_E 3
oov_I 3
oov_S 3
EOF

# Expecting the user to have done run.sh to have $model_dir,
# $sat_model_dir, $lang, $lang_test, $train_data_dir
local/segmentation/prepare_unsad_data.sh \
  --sad-map $dir/fisher_sad.map \
  --config-dir conf \
  --reco-nj 40 --nj 100 --cmd "$train_cmd" \
  --sat-model $sat_model_dir \
  --lang-test $lang_test \
  $train_data_dir $lang $model_dir $dir

orig_data_dir=${train_data_dir}_sp

data_dir=${train_data_dir}_whole

if [ ! -z $subset ]; then
  # Work on a subset
  utils/subset_data_dir.sh ${data_dir} $subset \
    ${data_dir}_$subset
  data_dir=${data_dir}_$subset
fi

reco_vad_dir=$dir/`basename $model_dir`_reco_vad_`basename $train_data_dir`_sp

# Add noise from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir.sh \
  --num-data-reps 5 \
  --data-dir $data_dir \
  --reco-vad-dir $reco_vad_dir
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf 

# Add music from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir_music.sh \
  --num-data-reps 5 \
  --data-dir $data_dir \
  --reco-vad-dir $reco_vad_dir
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf

if [ ! -z $utt_subset ]; then
  utils/subset_data_dir.sh ${orig_data_dir} $utt_subset \
    ${orig_data_dir}_`echo $utt_subset | perl -e 's/000$/k/'`
  orig_data_dir=${orig_data_dir}_`echo $utt_subset | perl -e 's/000$/k/'`
fi

# Add overlapping speech from $orig_data_dir/segments and create a new data directory
utt_vad_dir=$dir/`baseline $sat_model_dir`_ali_`basename $train_data_dir`_sp_vad_`basename $train_data_dir`_sp
local/segmentation/do_corruption_data_dir_overlapped_speech.sh \
  --nj 40 --cmd queue.pl \
  --num-data-reps 1 \
  --data-dir ${orig_data_dir} \
  --utt-vad-dir $utt_vad_dir

local/segmentation/prepare_unsad_overlapped_speech_labels.sh \
  --num-data-reps 1 --nj 40 --cmd queue.pl \
  ${orig_data_dir}_ovlp_corrupted_hires_bp \
  ${orig_data_dir}_ovlp_corrupted/overlapped_segments_info.txt \
  $utt_vad_dir exp/make_overlap_labels overlap_labels
