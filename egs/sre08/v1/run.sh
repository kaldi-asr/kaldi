#!/bin/bash
# Copyright 2013   Daniel Povey
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.
#
# Also see local/run_more_data.sh which duplicates this example but
# with additional development data.
# 
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
utils/combine_data.sh data/fisher data/fisher1 data/fisher2 

local/make_sre_2008_train.pl /export/corpora5/LDC/LDC2011S05 data
local/make_sre_2008_test.sh  /export/corpora5/LDC/LDC2011S08 data


mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

set -e
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/fisher exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/sre08_train_short2_female exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/sre08_train_short2_male exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/sre08_test_short3_female exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/sre08_test_short3_male exp/make_mfcc $mfccdir


sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/fisher exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/sre08_train_short2_female exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/sre08_train_short2_male exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/sre08_test_short3_female exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/sre08_test_short3_male exp/make_vad $vaddir


# Note: to see the proportion of voiced frames you can do,
# grep Prop exp/make_vad/vad_*.1.log 

# Get male and female subsets of training data.
grep -w m data/fisher/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/fisher data/fisher_male
grep -w f data/fisher/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/fisher data/fisher_female
rm foo

# Get smaller subsets of training data for faster training.
utils/subset_data_dir.sh data/fisher 2000 data/fisher_2k
utils/subset_data_dir.sh data/fisher 4000 data/fisher_4k
utils/subset_data_dir.sh data/fisher_male 4000 data/fisher_male_4k
utils/subset_data_dir.sh data/fisher_female 4000 data/fisher_female_4k


sid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/fisher_2k 2048 exp/diag_ubm_2048

sid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/fisher_4k exp/diag_ubm_2048 exp/full_ubm_2048

# Get male and female versions of the UBM in one pass; make sure not to remove
# any Gaussians due to low counts (so they stay matched).  This will be more convenient
# for gender-id.
sid/train_full_ubm.sh --nj 30 --remove-low-count-gaussians false --num-iters 1 --cmd "$train_cmd" \
   data/fisher_male_4k exp/full_ubm_2048 exp/full_ubm_2048_male &
sid/train_full_ubm.sh --nj 30 --remove-low-count-gaussians false --num-iters 1 --cmd "$train_cmd" \
   data/fisher_female_4k exp/full_ubm_2048 exp/full_ubm_2048_female &
wait

# note, the mem_free,ram_free is counted per thread... in this setup each
# job has 4 processes running each with 4 threads; each job takes about 5G
# of memory so we need about 20G, plus add memory for sum-accs to make it 25G.
# but we'll submit using -pe smp 16, and this multiplies the memory requirement
# by 16, so submitting with 2G as the requirement, to make the total requirement
# 32, is reasonable.

# Train the iVector extractor for male speakers.
sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=2G,ram_free=2G" \
  --num-iters 5 exp/full_ubm_2048_male/final.ubm data/fisher_male \
  exp/extractor_2048_male

# The same for female speakers.
sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=2G,ram_free=2G" \
  --num-iters 5 exp/full_ubm_2048_female/final.ubm data/fisher_female \
  exp/extractor_2048_female

# The script below demonstrates the gender-id script.  We don't really use
# it for anything here, because the SRE 2008 data is already split up by
# gender and gender identification is not required for the eval.
# It prints out the error rate based on the info in the spk2gender file;
# see exp/gender_id_fisher/error_rate where it is also printed.
sid/gender_id.sh --cmd "$train_cmd" --nj 150 exp/full_ubm_2048{,_male,_female} \
  data/fisher exp/gender_id_fisher
# Gender-id error rate is 2.58%

# Extract the iVectors for the Fisher data.
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_male data/fisher_male exp/ivectors_fisher_male

sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_female data/fisher_female exp/ivectors_fisher_female

# .. and for the SRE08 training and test data. (We focus on the main
# evaluation condition, the only required one in that eval, which is
# the short2-short3 eval.)
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_female data/sre08_train_short2_female exp/ivectors_sre08_train_short2_female
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_male data/sre08_train_short2_male exp/ivectors_sre08_train_short2_male
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_female data/sre08_test_short3_female exp/ivectors_sre08_test_short3_female
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_male data/sre08_test_short3_male exp/ivectors_sre08_test_short3_male

### Demonstrate simple cosine-distance scoring:

trials=data/sre08_trials/short2-short3-female.trials
cat $trials | awk '{print $1, $2}' | \
 ivector-compute-dot-products - \
  scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp \
  scp:exp/ivectors_sre08_test_short3_female/spk_ivector.scp \
   foo 

local/score_sre08.sh $trials foo

# Results for Female:
# Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  28.14   4.78  27.68  21.77  20.07  11.03   6.97   7.63
trials=data/sre08_trials/short2-short3-male.trials
cat $trials | awk '{print $1, $2}' | \
 ivector-compute-dot-products - \
  scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp \
  scp:exp/ivectors_sre08_test_short3_male/spk_ivector.scp \
   foo

local/score_sre08.sh $trials foo

# Results for Male:
# Scoring against data/sre08_trials/short2-short3-male.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  26.38   3.63  26.39  17.31  17.34   8.24   6.83   5.26

# The following shows a more direct way to get the scores.
#condition=6
#awk '{print $3}' foo | paste - $trials | awk -v c=$condition '{n=4+c; if ($n == "Y") print $1, $4}' | \
#  compute-eer -
# LOG (compute-eer:main():compute-eer.cc:136) Equal error rate is 11.1419%, at threshold 55.9827

# Note: to see how you can plot the DET curve, look at
# local/det_curve_example.sh


### Demonstrate what happens if we reduce the dimension with LDA

 ivector-compute-lda --dim=150  --total-covariance-factor=0.1 \
  'ark:ivector-normalize-length scp:exp/ivectors_fisher_female/ivector.scp  ark:- |' ark:data/fisher_female/utt2spk \
    exp/ivectors_fisher_female/transform.mat

 trials=data/sre08_trials/short2-short3-female.trials
 cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
   'ark:ivector-transform exp/ivectors_fisher_female/transform.mat scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   'ark:ivector-transform exp/ivectors_fisher_female/transform.mat scp:exp/ivectors_sre08_test_short3_female/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   foo

local/score_sre08.sh $trials foo

# Results for Female:
# Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  23.86   2.09  23.46  15.92  16.71  10.09   5.96   7.37

 ivector-compute-lda --dim=150 --total-covariance-factor=0.1 \
  'ark:ivector-normalize-length scp:exp/ivectors_fisher_male/ivector.scp  ark:- |' ark:data/fisher_male/utt2spk \
    exp/ivectors_fisher_male/transform.mat

 trials=data/sre08_trials/short2-short3-male.trials
 cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
   'ark:ivector-transform exp/ivectors_fisher_male/transform.mat scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   'ark:ivector-transform exp/ivectors_fisher_male/transform.mat scp:exp/ivectors_sre08_test_short3_male/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   foo

local/score_sre08.sh $trials foo

# Results for Male:
# Scoring against data/sre08_trials/short2-short3-male.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  18.96   1.61  18.93  12.30  12.50   8.01   6.83   4.82

### Demonstrate PLDA scoring:

## Note: below, the ivector-subtract-global-mean step doesn't appear to affect
## the EER, although it does shift the threshold.

 trials=data/sre08_trials/short2-short3-female.trials
 cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
   'ark:ivector-transform exp/ivectors_fisher_female/transform.mat scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   'ark:ivector-transform exp/ivectors_fisher_female/transform.mat scp:exp/ivectors_sre08_test_short3_female/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   foo

ivector-compute-plda ark:data/fisher_female/spk2utt \
  'ark:ivector-normalize-length scp:exp/ivectors_fisher_female/ivector.scp  ark:- |' \
    exp/ivectors_fisher_female/plda 2>exp/ivectors_fisher_female/log/plda.log


ivector-plda-scoring --num-utts=ark:exp/ivectors_sre08_train_short2_female/num_utts.ark \
   "ivector-copy-plda --smoothing=0.0 exp/ivectors_fisher_female/plda - |" \
   "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp ark:- |" \
   "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_test_short3_female/ivector.scp ark:- |" \
   "cat '$trials' | awk '{print \$1, \$2}' |" foo

local/score_sre08.sh $trials foo

# Result for Female is below:
# Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  20.03   2.09  20.21  16.67  12.14   8.59   4.82   5.00

 trials=data/sre08_trials/short2-short3-male.trials
 cat $trials | awk '{print $1, $2}' | \
  ivector-compute-dot-products - \
   'ark:ivector-transform exp/ivectors_fisher_male/transform.mat scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   'ark:ivector-transform exp/ivectors_fisher_male/transform.mat scp:exp/ivectors_sre08_test_short3_male/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |' \
   foo

ivector-compute-plda ark:data/fisher_male/spk2utt \
  'ark:ivector-normalize-length scp:exp/ivectors_fisher_male/ivector.scp  ark:- |' \
    exp/ivectors_fisher_male/plda 2>exp/ivectors_fisher_male/log/plda.log

ivector-plda-scoring --num-utts=ark:exp/ivectors_sre08_train_short2_male/num_utts.ark \
   "ivector-copy-plda --smoothing=0.0 exp/ivectors_fisher_male/plda - |" \
   "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp ark:- |" \
   "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_test_short3_male/ivector.scp ark:- |" \
   "cat '$trials' | awk '{print \$1, \$2}' |" foo

local/score_sre08.sh $trials foo

# Result for Male is below:
# Scoring against data/sre08_trials/short2-short3-male.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  15.98   1.61  16.27  12.98  11.25   6.75   4.78   4.39
