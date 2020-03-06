#!/usr/bin/env bash

# This script removes non-speech, long musics or long silence from the original
# speech recordings.

nj=32
stage=0
cmd=run.pl
. cmd.sh
. path.sh
. utils/parse_options.sh

set -e -o pipefail
if [ $# -ne 5 ]; then
    echo "Usage: $0 <src-data-dir> <lang-dir> <mdl-dir> <dir> <cleaned-data>"
    echo "E.g., $0 [options] data/train data/lang <gale_mandir_mdl_dir> exp/gale_mandarin data/train_clean"
    exit 1;
fi

src_data_dir=$1
lang_dir=$2
mdldir=$3
newdir=$4
clean_data_dir=$5

steps/cleanup/segment_long_utterances.sh --nj ${nj} --cmd "$train_cmd" --stage $stage \
  --max-bad-proportion 0.6 $mdldir $lang_dir $src_data_dir \
  $clean_data_dir $newdir || exit 1;

echo "Clean up succeeded !"

