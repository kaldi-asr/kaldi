#! /bin/bash

# This script prepares Fisher data for training a speech activity detection 
# and music detection system

# Copyright 2016  Vimal Manohar
# Apache 2.0.

. path.sh
. cmd.sh

set -e -o pipefail

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  echo "This script is to serve as an example recipe."
  echo "Edit the script to change variables if needed."
  exit 1
fi

subset_fraction=0.15
realign=false

# All the paths below can be modified to any absolute path.
ROOT_DIR=/export/a15/vmanoha1/workspace_snr/egs/aspire/s5

stage=-1

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  echo "This script is to serve as an example recipe."
  echo "Edit the script to change variables if needed."
  exit 1
fi

dir=exp/unsad_simple/make_unsad_fisher_train_100k_sp  # Work dir
train_data_dir=$ROOT_DIR/data/train_100k_sp
unperturbed_data_dir=$ROOT_DIR/data/train_100k
model_dir=$ROOT_DIR/exp/tri4a
lang=$ROOT_DIR/data/lang  # Language directory

mkdir -p $dir

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

if [ ! -d RIRS_NOISES/ ]; then
  # Prepare MUSAN rirs and noises
  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  unzip rirs_noises.zip
fi

if [ ! -d RIRS_NOISES/music ]; then
  # Prepare MUSAN music
  local/segmentation/prepare_musan_music.sh /export/corpora/JHU/musan RIRS_NOISES/music
fi

utils/copy_data_dir.sh $train_data_dir data/fisher_train_100k_simple_sp
train_data_dir=data/fisher_train_100k_simple_sp

utils/copy_data_dir.sh $unperturbed_data_dir data/fisher_train_100k_simple
unperturbed_data_dir=data/fisher_train_100k_simple

# Expecting the user to have done run.sh to have $model_dir,
# $sat_model_dir, $lang, $lang_test, $train_data_dir
if $realign; then
  ali_dir=$dir/`basename $model_dir`_ali_$(basename $train_data_dir)

  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    $train_data_dir $lang $model_dir $ali_dir

  local/segmentation/prepare_unsad_data_simple.sh \
    --sad-map $dir/fisher_sad.map --cmd "$train_cmd" \
    $train_data_dir $lang $ali_dir $dir

  vad_dir=$dir/`basename $ali_dir`_vad_$(basename $train_data_dir)
else
  local/segmentation/prepare_unsad_data_simple.sh --speed-perturb true \
    --sad-map $dir/fisher_sad.map --cmd "$train_cmd" \
    $unperturbed_data_dir $lang $model_dir $dir

  vad_dir=$dir/`basename $model_dir`_vad_$(basename $unperturbed_data_dir)
fi

data_dir=${unperturbed_data_dir}

if [ ! -z "$subset_fraction" ]; then
  # Work on a subset
  num_utts=`cat $unperturbed_data_dir/utt2spk | wc -l`
  subset=`python -c "n=int($num_utts * $subset_fraction / 1000.0) * 1000; print (n if n > 4000 else 4000)"`
  subset_affix=`echo $subset | perl -pe 's/000/k/g'`
  utils/subset_data_dir.sh --speakers ${unperturbed_data_dir} $subset \
    ${unperturbed_data_dir}_${subset_affix}
  data_dir=${unperturbed_data_dir}_${subset_affix}
fi

# Add noise from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir_snr.sh \
  --cmd "$train_cmd" --nj 40 --stage 8 \
  --data-dir $data_dir \
  --vad-dir $vad_dir \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf 

# Add music from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir_music.sh \
  --cmd "$train_cmd" --nj 40 \
  --data-dir $data_dir \
  --vad-dir $vad_dir \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf
