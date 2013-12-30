#!/bin/bash

# Just for the female models, running without regression of the
# log-weights on the iVector.  See if the weights were helping.

sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=2G,ram_free=2G" \
  --use-weights false --num-iters 5 exp/full_ubm_2048_female/final.ubm data/fisher_female \
  exp/extractor_2048_female_noweights

sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_female_noweights data/sre08_train_short2_female exp/ivectors_sre08_train_short2_female_noweights

sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_female_noweights data/sre08_test_short3_female exp/ivectors_sre08_test_short3_female_noweights

trials=data/sre08_trials/short2-short3-female.trials
cat $trials | awk '{print $1, $2}' | \
 ivector-compute-dot-products - \
  scp:exp/ivectors_sre08_train_short2_female_noweights/spk_ivector.scp \
  scp:exp/ivectors_sre08_test_short3_female_noweights/spk_ivector.scp \
   foo

local/score_sre08.sh $trials foo

# Scores were:
#Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  27.75   4.48  27.36  21.77  19.95  10.70   7.22   7.37

# Baseline (using the weights) was below.  Baseline is worse on conditions
#  1, 3, 6, 7, 8, same on 2, better on 4, 5.
# So it doesn't lead to any great confidence that the weights were helping.
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  29.07   4.48  28.89  20.57  19.83  11.14   7.35   7.89


# will now try PLDA, etc.
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048_female_noweights data/fisher_female exp/ivectors_fisher_female_noweights


ivector-compute-plda ark:data/fisher_female/spk2utt \
  'ark:ivector-normalize-length scp:exp/ivectors_fisher_female_noweights/ivector.scp  ark:- |' \
    exp/ivectors_fisher_female_noweights/plda 2>exp/ivectors_fisher_female_noweights/log/plda.log


ivector-plda-scoring --num-utts=ark:exp/ivectors_sre08_train_short2_female_noweights/num_utts.ark \
   "ivector-copy-plda --smoothing=0.0 exp/ivectors_fisher_female_noweights/plda - |" \
   "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_train_short2_female_noweights/spk_ivector.scp ark:- |" \
   "ark:ivector-subtract-global-mean scp:exp/ivectors_sre08_test_short3_female_noweights/ivector.scp ark:- |" \
   "cat '$trials' | awk '{print \$1, \$2}' |" foo

local/score_sre08.sh $trials foo

# Result is below:
#Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  19.66   2.69  19.72  17.87  12.26   8.09   4.56   4.47


# Baseline (with weights) is below.  Without weights is better on conditions
# 1, 3, 6, 7, 8. 
# With weights is better on conditions 2, 4, 5.
# 
#Scoring against data/sre08_trials/short2-short3-female.trials
#  Condition:      1      2      3      4      5      6      7      8
#        EER:  20.55   2.09  20.76  17.27  12.14   8.59   4.69   4.74

