#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares Babel data for training speech activity detection,
# music detection.

. path.sh
. cmd.sh

set -e
set -o pipefail
set -u

lang_id=assamese
subset=     # Number of recordings to keep before speed perturbation and corruption. 
            # In limitedLP, this is about 120. So subset, if specified, must be lower that that.

# All the paths below can be modified to any absolute path.
ROOT_DIR=/home/vimal/workspace_waveform/egs/babel/s5c_assamese/

stage=-1

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  echo "This script is to serve as an example recipe."
  echo "Edit the script to change variables if needed."
  exit 1
fi

dir=exp/unsad/make_unsad_babel_${lang_id}_train   # Work dir

model_dir=$ROOT_DIR/exp/tri4  # Model directory used for decoding
sat_model_dir=$ROOT_DIR/exp/tri5   # Model directory used for getting alignments
lang=$ROOT_DIR/data/lang  # Language directory
lang_test=$ROOT_DIR/data/lang  # Language directory used to build graph

mkdir -p $dir

# Hard code the mapping from phones to SAD labels
# 0 for silence, 1 for speech, 2 for noise, 3 for unk
cat <<EOF > $dir/babel_sad.map
<oov> 3
<oov>_B 3
<oov>_E 3
<oov>_I 3
<oov>_S 3
<sss> 2
<sss>_B 2
<sss>_E 2
<sss>_I 2
<sss>_S 2
<vns> 2
<vns>_B 2
<vns>_E 2
<vns>_I 2
<vns>_S 2
SIL 0
SIL_B 0
SIL_E 0
SIL_I 0
SIL_S 0
EOF

# The original data directory which will be converted to a whole (recording-level) directory.
utils/copy_data_dir.sh $ROOT_DIR/data/train data/babel_${lang_id}_train
train_data_dir=data/babel_${lang_id}_train

# Expecting the user to have done run.sh to have $model_dir,
# $sat_model_dir, $lang, $lang_test, $train_data_dir
local/segmentation/prepare_unsad_data.sh --stage 14 \
  --sad-map $dir/babel_sad.map \
  --config-dir $ROOT_DIR/conf --feat-type plp --add-pitch true \
  --reco-nj 40 --nj 100 --cmd "$train_cmd" \
  --sat-model-dir $sat_model_dir \
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
  --data-dir $data_dir \
  --reco-vad-dir $reco_vad_dir \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf 

# Add music from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir_music.sh \
  --data-dir $data_dir \
  --reco-vad-dir $reco_vad_dir \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf
