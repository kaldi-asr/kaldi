#!/bin/bash
# Copyright 2013   Daniel Povey
#      2014-2016   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

# This example script is still a bit of a mess, and needs to be
# cleaned up, but it shows you all the basic ingredients.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

local/make_fisher.sh /export/corpora3/LDC/{LDC2004S13,LDC2004T19} data/fisher1
#Processed 4948 utterances; 902 had missing wav data. (note: we should figure
#out why so much data goes missing.)
local/make_fisher.sh /export/corpora3/LDC/{LDC2005S13,LDC2005T19} data/fisher2
#Processed 5848 utterances; 1 had missing wav data.

local/make_sre_2005_test.pl /export/corpora5/LDC/LDC2011S04 data
local/make_sre_2004_test.pl \
  /export/corpora5/LDC/LDC2006S44/r93_5_1/sp04-05/test data/sre_2004_1
local/make_sre_2004_test.pl \
  /export/corpora5/LDC/LDC2006S44/r93_6_1/sp04-06/test data/sre_2004_2
local/make_sre_2008_train.pl /export/corpora5/LDC/LDC2011S05 data
local/make_sre_2008_test.sh  /export/corpora5/LDC/LDC2011S08 data
local/make_sre_2006_train.pl /export/corpora5/LDC/LDC2011S09 data
local/make_sre_2005_train.pl /export/corpora5/LDC/LDC2011S01 data
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
  data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
  data/swbd_cellular2_train

utils/combine_data.sh data/train data/fisher1 data/fisher2 \
  data/swbd_cellular1_train data/swbd_cellular2_train \
  data/sre05_train_3conv4w_female data/sre05_train_8conv4w_female \
  data/sre06_train_3conv4w_female data/sre06_train_8conv4w_female \
  data/sre05_train_3conv4w_male data/sre05_train_8conv4w_male \
  data/sre06_train_3conv4w_male data/sre06_train_8conv4w_male \
  data/sre_2004_1/ data/sre_2004_2/ data/sre05_test

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

set -e
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre08_train_short2_female exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre08_train_short2_male exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre08_test_short3_female exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre08_test_short3_male exp/make_mfcc $mfccdir


sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
  data/train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
  data/sre08_train_short2_female exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
  data/sre08_train_short2_male exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
  data/sre08_test_short3_female exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
  data/sre08_test_short3_male exp/make_vad $vaddir


# Note: to see the proportion of voiced frames you can do,
# grep Prop exp/make_vad/vad_*.1.log

# Get male and female subsets of training data.
grep -w m data/train/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/train data/train_male
grep -w f data/train/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/train data/train_female
rm foo

# Get smaller subsets of training data for faster training.
utils/subset_data_dir.sh data/train 4000 data/train_4k
utils/subset_data_dir.sh data/train 8000 data/train_8k
utils/subset_data_dir.sh data/train_male 8000 data/train_male_8k
utils/subset_data_dir.sh data/train_female 8000 data/train_female_8k

# The recipe currently uses delta-window=3 and delta-order=2. However
# the accuracy is almost as good using delta-window=4 and delta-order=1
# and could be faster due to lower dimensional features.  Alternative
# delta options (e.g., --delta-window 4 --delta-order 1) can be provided to
# sid/train_diag_ubm.sh.  The options will be propagated to the other scripts.
sid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/train_4k 2048 \
  exp/diag_ubm_2048

sid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/train_8k \
  exp/diag_ubm_2048 exp/full_ubm_2048

# Get male and female versions of the UBM in one pass; make sure not to remove
# any Gaussians due to low counts (so they stay matched).  This will be
# more convenient for gender-id.
sid/train_full_ubm.sh --nj 30 --remove-low-count-gaussians false \
  --num-iters 1 --cmd "$train_cmd" \
  data/train_male_8k exp/full_ubm_2048 exp/full_ubm_2048_male &
sid/train_full_ubm.sh --nj 30 --remove-low-count-gaussians false \
  --num-iters 1 --cmd "$train_cmd" \
  data/train_female_8k exp/full_ubm_2048 exp/full_ubm_2048_female &
wait

# Train the iVector extractor for male speakers.
sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --num-iters 5 exp/full_ubm_2048_male/final.ubm data/train_male \
  exp/extractor_2048_male

# The same for female speakers.
sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --num-iters 5 exp/full_ubm_2048_female/final.ubm data/train_female \
  exp/extractor_2048_female

# The script below demonstrates the gender-id script.  We don't really use
# it for anything here, because the SRE 2008 data is already split up by
# gender and gender identification is not required for the eval.
# It prints out the error rate based on the info in the spk2gender file;
# see exp/gender_id_fisher/error_rate where it is also printed.
sid/gender_id.sh --cmd "$train_cmd" --nj 150 exp/full_ubm_2048{,_male,_female} \
  data/train exp/gender_id_train
# Gender-id error rate is 3.41%

# Extract the iVectors for the training data.
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor_2048_male data/train_male exp/ivectors_train_male

sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor_2048_female data/train_female exp/ivectors_train_female

# .. and for the SRE08 training and test data. (We focus on the main
# evaluation condition, the only required one in that eval, which is
# the short2-short3 eval.)
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor_2048_female data/sre08_train_short2_female \
  exp/ivectors_sre08_train_short2_female
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor_2048_male data/sre08_train_short2_male \
  exp/ivectors_sre08_train_short2_male
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor_2048_female data/sre08_test_short3_female \
  exp/ivectors_sre08_test_short3_female
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor_2048_male data/sre08_test_short3_male \
  exp/ivectors_sre08_test_short3_male

### Demonstrate simple cosine-distance scoring:

trials=data/sre08_trials/short2-short3-female.trials

# Note: speaker-level i-vectors have already been length-normalized
# by sid/extract_ivectors.sh, but the utterance-level test i-vectors
# have not.
cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
  scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp \
  'ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_female/ivector.scp ark:- |' \
  foo
local/score_sre08.sh $trials foo

# Results for Female:
# Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:  12.70  20.09   4.78  19.08  16.37  15.87  10.42   7.10   7.89

trials=data/sre08_trials/short2-short3-male.trials
cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
  scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp \
  'ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_male/ivector.scp ark:- |' \
  foo
local/score_sre08.sh $trials foo

# Results for Male:
# Scoring against data/sre08_trials/short2-short3-male.trials
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:  11.10  18.55   5.24  18.03  14.35  13.44   8.47   5.92   4.82

# The following shows a more direct way to get the scores.
# condition=6
# awk '{print $3}' foo | paste - $trials | awk -v c=$condition '{n=4+c; \\
# if ($n == "Y") print $1, $4}' | \
#  compute-eer -
# LOG (compute-eer:main():compute-eer.cc:136) Equal error rate is 11.10%,
# at threshold 55.9827

# Note: to see how you can plot the DET curve, look at
# local/det_curve_example.sh


### Demonstrate what happens if we reduce the dimension with LDA

ivector-compute-lda --dim=150  --total-covariance-factor=0.1 \
  'ark:ivector-normalize-length scp:exp/ivectors_train_female/ivector.scp ark:- |' \
  ark:data/train_female/utt2spk \
  exp/ivectors_train_female/transform.mat

trials=data/sre08_trials/short2-short3-female.trials
cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
  'ark:ivector-transform exp/ivectors_train_female/transform.mat scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
  'ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_female/ivector.scp ark:- | ivector-transform exp/ivectors_train_female/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |' \
  foo
local/score_sre08.sh $trials foo

# Results for Female:
# Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   7.96   9.82   1.49   9.44  10.51  10.70   8.81   5.83   7.11

ivector-compute-lda --dim=150 --total-covariance-factor=0.1 \
  'ark:ivector-normalize-length scp:exp/ivectors_train_male/ivector.scp ark:- |' \
  ark:data/train_male/utt2spk \
  exp/ivectors_train_male/transform.mat

trials=data/sre08_trials/short2-short3-male.trials
  cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
  'ark:ivector-transform exp/ivectors_train_male/transform.mat scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
  'ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_male/ivector.scp ark:- | ivector-transform exp/ivectors_train_male/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |' \
  foo
local/score_sre08.sh $trials foo

# Results for Male:
# Scoring against data/sre08_trials/short2-short3-male.trials
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   6.20   8.30   1.21   8.10   8.43   7.03   7.32   5.70   3.51


### Demonstrate PLDA scoring:

## Note: below, the ivector-subtract-global-mean step doesn't appear to affect
## the EER, although it does shift the threshold.

trials=data/sre08_trials/short2-short3-female.trials
ivector-compute-plda ark:data/train_female/spk2utt \
  'ark:ivector-normalize-length scp:exp/ivectors_train_female/ivector.scp  ark:- |' \
  exp/ivectors_train_female/plda 2>exp/ivectors_train_female/log/plda.log
ivector-plda-scoring --simple-length-normalization=true --num-utts=ark:exp/ivectors_sre08_train_short2_female/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_female/plda - |" \
  "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp ark:- |" \
  "ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_female/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
  "cat '$trials' | awk '{print \$1, \$2}' |" foo
local/score_sre08.sh $trials foo

# Result for Female is below:
# Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   6.44   9.76   1.49   9.76   7.66   7.21   6.87   4.06   4.74

trials=data/sre08_trials/short2-short3-male.trials
ivector-compute-plda ark:data/train_male/spk2utt \
  'ark:ivector-normalize-length scp:exp/ivectors_train_male/ivector.scp  ark:- |' \
  exp/ivectors_train_male/plda 2>exp/ivectors_train_male/log/plda.log
ivector-plda-scoring --simple-length-normalization=true --num-utts=ark:exp/ivectors_sre08_train_short2_male/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_male/plda - |" \
  "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp ark:- |" \
  "ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_male/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:- |" \
  "cat '$trials' | awk '{print \$1, \$2}' |" foo; local/score_sre08.sh $trials foo

# Result for Male is below:
# Scoring against data/sre08_trials/short2-short3-male.trials
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   4.68   7.41   1.21   7.48   5.70   4.69   5.61   3.19   2.19


### Demonstrate PLDA scoring after adapting the out-of-domain PLDA model with in-domain training data:

# first, female.
trials=data/sre08_trials/short2-short3-female.trials
cat exp/ivectors_sre08_train_short2_female/spk_ivector.scp exp/ivectors_sre08_test_short3_female/ivector.scp > female.scp
ivector-plda-scoring --simple-length-normalization=true --num-utts=ark:exp/ivectors_sre08_train_short2_female/num_utts.ark \
  "ivector-adapt-plda $adapt_opts exp/ivectors_train_female/plda scp:female.scp -|" \
  scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp \
  "ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_female/ivector.scp ark:- |" \
  "cat '$trials' | awk '{print \$1, \$2}' |" foo; local/score_sre08.sh $trials foo
# Results:
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   5.45   6.73   1.19   6.79   7.06   6.61   6.32   4.18   4.74
# Baseline (repeated from above):
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   6.44   9.76   1.49   9.76   7.66   7.21   6.87   4.06   4.74

# next, male.
trials=data/sre08_trials/short2-short3-male.trials
cat exp/ivectors_sre08_train_short2_male/spk_ivector.scp exp/ivectors_sre08_test_short3_male/ivector.scp > male.scp
ivector-plda-scoring --simple-length-normalization=true --num-utts=ark:exp/ivectors_sre08_train_short2_male/num_utts.ark \
  "ivector-adapt-plda $adapt_opts exp/ivectors_train_male/plda scp:male.scp -|" \
  scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp \
  "ark:ivector-normalize-length scp:exp/ivectors_sre08_test_short3_male/ivector.scp ark:- |" \
  "cat '$trials' | awk '{print \$1, \$2}' |" foo; local/score_sre08.sh $trials foo

# Results:
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   4.03   4.71   0.81   4.73   5.01   4.84   5.61   3.87   2.63
# Baseline is as follows, repeated from above.  Focus on condition 0 (= all).
#  Condition:      0      1      2      3      4      5      6      7      8
#        EER:   4.68   7.41   1.21   7.48   5.70   4.69   5.61   3.19   2.19
