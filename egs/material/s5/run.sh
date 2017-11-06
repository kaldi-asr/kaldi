#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
. ./path.sh
. ./cmd.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

corpus=/export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/
#corpus=/export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1B-BUILD_v1.0/

local/prepare_text_data.sh $corpus
local/prepare_audio_data.sh $corpus

utils/fix_data_dir.sh data/train
steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 data/train
utils/fix_data_dir.sh data/train
utils/validate_data_dir.sh data/train

utils/fix_data_dir.sh data/dev
steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 data/dev
utils/fix_data_dir.sh data/dev
utils/validate_data_dir.sh data/dev

local/prepare_dict.sh $corpus
utils/validate_dict_dir.pl data/local/dict
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
utils/validate_lang.pl data/lang

local/train_lms_srilm.sh --oov-symbol "<unk>" data data/lm
utils/format_lm.sh data/lang data/lm/lm.gz \
  data/local/dict/lexiconp.txt data/lang_test
utils/validate_lang.pl data/lang_test

