#!/usr/bin/env bash
# Copyright  2016-2017  Go-Vivace Inc. (Author: Mousmita Sarma)
#
# Apache 2.0.
#
# This script runs the NIST 2007 General Language Recognition Closed-Set
# evaluation.
# This example script shows how to replace the GMM-UBM
# with a DNN trained for ASR.

. ./cmd.sh
. ./path.sh
set -e

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
languages=local/general_lr_closed_set_langs.txt
nnet=exp/nnet2_online/nnet_ms_a/final.mdl

# Train a DNN on about 1800 hours of the english portion of Fisher.
local/dnn/train_dnn.sh

data_root=/export/corpora/LDC
# Training data sources
local/make_sre_2008_train.pl $data_root/LDC2011S05 data
local/make_callfriend.pl $data_root/LDC96S60 vietnamese data
local/make_callfriend.pl $data_root/LDC96S59 tamil data
local/make_callfriend.pl $data_root/LDC96S53 japanese data
local/make_callfriend.pl $data_root/LDC96S52 hindi data
local/make_callfriend.pl $data_root/LDC96S51 german data
local/make_callfriend.pl $data_root/LDC96S50 farsi data
local/make_callfriend.pl $data_root/LDC96S48 french data
local/make_callfriend.pl $data_root/LDC96S49 arabic.standard data
local/make_callfriend.pl $data_root/LDC96S54 korean data
local/make_callfriend.pl $data_root/LDC96S55 chinese.mandarin.mainland data
local/make_callfriend.pl $data_root/LDC96S56 chinese.mandarin.taiwan data
local/make_callfriend.pl $data_root/LDC96S57 spanish.caribbean data
local/make_callfriend.pl $data_root/LDC96S58 spanish.noncaribbean data
local/make_lre03.pl $data_root/LDC/LDC2006S31 data
local/make_lre05.pl $data_root/LDC/LDC2008S05 data
local/make_lre07_train.pl $data_root/LDC2009S05 data
local/make_lre09.pl /export/corpora5/NIST/LRE/LRE2009/eval data

# Make the evaluation data set. We're concentrating on the General Language
# Recognition Closet-Set evaluation, so we remove the dialects and filter
# out the unknown languages used in the open-set evaluation.
local/make_lre07.pl $data_root/LDC2009S04 data/lre07_all

cp -r data/lre07_all data/lre07
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/lre07_all/utt2lang) \
  > data/lre07/utt2lang
utils/fix_data_dir.sh data/lre07

src_list="data/sre08_train_10sec_female \
  data/sre08_train_10sec_male data/sre08_train_3conv_female \
  data/sre08_train_3conv_male data/sre08_train_8conv_female \
  data/sre08_train_8conv_male data/sre08_train_short2_male \
  data/sre08_train_short2_female data/ldc96* data/lid05d1 \
  data/lid05e1 data/lid96d1 data/lid96e1 data/lre03 \
  data/ldc2009* data/lre09"
# Remove any spk2gender files that we have: since not all data
# sources have this info, it will cause problems with combine_data.sh
for d in $src_list; do rm -f $d/spk2gender 2>/dev/null; done

utils/combine_data.sh data/train_unsplit $src_list

# original utt2lang will remain in data/train_unsplit/.backup/utt2lang.
utils/apply_map.pl -f 2 --permissive local/lang_map.txt  < data/train_unsplit/utt2lang  2>/dev/null > foo
cp foo data/train_unsplit/utt2lang
rm foo

local/split_long_utts.sh --max-utt-len 120 data/train_unsplit data/train

echo "**Language count in i-Vector extractor training (after splitting long utterances):**"
awk '{print $2}' data/train/utt2lang | sort | uniq -c | sort -nr

use_vtln=true
if $use_vtln; then
  for t in train lre07; do
    cp -r data/${t} data/${t}_novtln
    rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
    steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 12 --cmd "$train_cmd" \
       data/${t}_novtln exp/make_mfcc $mfccdir
    lid/compute_vad_decision.sh data/${t}_novtln exp/make_mfcc $mfccdir
  done

  # Vtln-related things:
  # We'll use a subset of utterances to train the GMM we'll use for VTLN
  # warping.
  utils/subset_data_dir.sh data/train_novtln 5000 data/train_5k_novtln

  # Note, we're using the speaker-id version of the train_diag_ubm.sh script, which
  # uses double-delta instead of SDC features to train a 256-Gaussian UBM.
  sid/train_diag_ubm.sh --nj 12 --cmd "$train_cmd" data/train_5k_novtln 256 \
    exp/diag_ubm_vtln
  lid/train_lvtln_model.sh --mfcc-config conf/mfcc_vtln.conf --nj 12 --cmd "$train_cmd" \
     data/train_5k_novtln exp/diag_ubm_vtln exp/vtln

  for t in lre07 train; do
    lid/get_vtln_warps.sh --nj 12 --cmd "$train_cmd" \
       data/${t}_novtln exp/vtln exp/${t}_warps
    cp exp/${t}_warps/utt2warp data/$t/
  done

  utils/fix_data_dir.sh data/train
  utils/filter_scp.pl data/train/utt2warp data/train/utt2spk > data/train/utt2spk_tmp
  cp data/train/utt2spk_tmp data/train/utt2spk
  utils/fix_data_dir.sh data/train
fi

cp -r data/train data/train_dnn
cp -r data/lre07 data/lre07_dnn

# Extract language recognition features
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 12 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 12 --cmd "$train_cmd" \
  data/lre07 exp/make_mfcc $mfccdir

# Extract DNN features
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 12 --cmd "$train_cmd" \
  data/train_dnn exp/make_mfcc $mfccdir

steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 12 --cmd "$train_cmd" \
  data/lre07_dnn exp/make_mfcc $mfccdir

for name in lre07_dnn train_dnn lre07 train; do
  utils/fix_data_dir.sh data/${name}
done

lid/compute_vad_decision.sh --nj 12 --cmd "$train_cmd" data/train \
  exp/make_vad $vaddir
lid/compute_vad_decision.sh --nj 12 --cmd "$train_cmd" data/lre07 \
  exp/make_vad $vaddir

for name in train lre07; do
  cp data/${name}/vad.scp data/${name}_dnn/vad.scp
  cp data/${name}/utt2spk data/${name}_dnn/utt2spk
  cp data/${name}/spk2utt data/${name}_dnn/spk2utt
  utils/fix_data_dir.sh data/${name}
  utils/fix_data_dir.sh data/${name}_dnn
done

# Subset training data for faster sup-GMM initialization.
utils/subset_data_dir.sh data/train 32000 data/train_32k
utils/fix_data_dir.sh data/train_32k
utils/subset_data_dir.sh data/train_dnn 32000 data/train_dnn_32k
utils/fix_data_dir.sh data/train_dnn_32k

# Initialize a full GMM from the DNN posteriors and language recognition
# features. This can be used both alone, as a UBM, or to initialize the
# i-vector extractor in a DNN-based system.
lid/init_full_ubm_from_dnn.sh --nj 8 --cmd "$train_cmd --mem 6G" \
  data/train_32k \
  data/train_dnn_32k $nnet exp/full_ubm

# Train an i-vector extractor based on the DNN-UBM.
lid/train_ivector_extractor_dnn.sh \
  --cmd "$train_cmd --mem 80G" --nnet-job-opt "--mem 4G" \
  --min-post 0.015 \
  --ivector-dim 600 \
  --num-iters 5 \
  --nj 5 exp/full_ubm/final.ubm $nnet \
  data/train \
  data/train_dnn \
  exp/extractor_dnn

# Filter out the languages we don't need for the closed-set eval
cp -r data/train data/train_lr
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/train/utt2lang) \
  > data/train_lr/utt2lang
utils/fix_data_dir.sh data/train_lr

echo "**Language count for logistic regression training (after splitting long utterances):**"
awk '{print $2}' data/train_lr/utt2lang | sort | uniq -c | sort -nr

cp -r data/train_dnn data/train_lr_dnn
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/train_dnn/utt2lang) \
  > data/train_lr_dnn/utt2lang
utils/fix_data_dir.sh data/train_lr_dnn

echo "**Language count for logistic regression training (after splitting long utterances):**"
awk '{print $2}' data/train_lr_dnn/utt2lang | sort | uniq -c | sort -nr

# Extract i-vectors using the extractor with the DNN-UBM
lid/extract_ivectors_dnn.sh --cmd "$train_cmd --mem 30G" \
  --nj 5 exp/extractor_dnn \
  $nnet \
  data/train_lr \
  data/train_lr_dnn \
  exp/ivectors_train

lid/extract_ivectors_dnn.sh --cmd "$train_cmd --mem 30G" \
  --nj 5 exp/extractor_dnn \
  $nnet \
  data/lre07 \
  data/lre07_dnn \
  exp/ivectors_lre07

# Train a logistic regression model on top of i-Vectors
lid/run_logistic_regression.sh --prior-scale 0.70 \
  --conf conf/logistic-regression.conf

# General LR 2007 closed-set eval
local/lre07_eval/lre07_eval.sh exp/ivectors_lre07 \
  local/general_lr_closed_set_langs.txt

#Duration (sec):    avg      3     10     30
#        ER (%):  16.18  31.43  12.38   4.73
#     C_avg (%):  10.27  19.67   7.84   3.31
