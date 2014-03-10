#!/bin/bash
# Copyright  2014   David Snyder
# Apache 2.0.
#
# An incomplete run.sh for this example. Currently this only trains up up a gender 
# independent UBM and ivector with the SRE08 training data.

. cmd.sh
. path.sh
set -e

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


local/make_sre_2008_train.pl /export/corpora5/LDC/LDC2011S05 data

local/make_callfriend.pl /export/corpora5/LDC/LDC96S48 french data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S49 arabic.standard data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S54 korean data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S55 chinese.mandarin.mainland data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S56 chinese.mandarin.taiwan data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S57 spanish.caribbean data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S58 spanish.noncaribbean data

local/make_lre03.pl /export/corpora4/LDC/LDC2006S31 data
local/make_lre05.pl /export/corpora5/LDC/LDC2008S05 data
local/make_lre07.pl /export/corpora5/LDC/LDC2009S04 data/lre07

src_list="data/sre08_train_10sec_female \
    data/sre08_train_10sec_male data/sre08_train_3conv_female \
    data/sre08_train_3conv_male data/sre08_train_8conv_female \
    data/sre08_train_8conv_male data/sre08_train_short2_male \
    data/sre08_train_short2_female data/ldc96s* data/lid05d1 \
    data/lid05e1 data/lid96d1 data/lid96e1 data/lre03"

# Remove any spk2gender files that we have: since not all data
# sources have this info, it will cause problems with combine_data.sh
for d in $src_list; do rm $d/spk2gender 2>/dev/null; done

utils/combine_data.sh data/train $src_list

# original utt2lang will remain in data/train/.backup/utt2lang.
utils/apply_map.pl -f 2 --permissive local/lang_map.txt  < data/train/utt2lang  2>/dev/null > foo
cp foo data/train/utt2lang
echo "**Language count in training:**"
awk '{print $2}' foo | sort | uniq -c | sort -nr
rm foo

##
## HERE
##

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 100 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/lre07 exp/make_mfcc $mfccdir

lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/train \
  exp/make_vad $vaddir
lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/lre07 \
  exp/make_vad $vaddir

# Use 4k of the 14k utterances for testing, but make sure the speakers do not
# overlap with the rest of the data, which will be used for training.
#utils/subset_data_dir.sh --speakers data/all 4000 data/lre07
#utils/filter_scp.pl --exclude data/lre07/spk2utt < data/all/spk2utt  | awk '{print $1}' > foo
#utils/subset_data_dir.sh --spk-list foo data/all data/train


utils/subset_data_dir.sh data/train 3000 data/train_3k
utils/subset_data_dir.sh data/train 6000 data/train_6k


lid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/train_3k 2048 \
  exp/diag_ubm_2048
lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/train_6k \
  exp/diag_ubm_2048 exp/full_ubm_2048_6k

lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/train \
  exp/full_ubm_2048_6k exp/full_ubm_2048


lid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=2G,ram_free=2G" \
  --num-iters 5 exp/full_ubm_2048/final.ubm data/train \
  exp/extractor_2048

lid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048 data/train exp/ivectors_train

lid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048 data/lre07 exp/ivectors_test


