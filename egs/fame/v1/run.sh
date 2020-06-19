#!/usr/bin/env bash
# Copyright 2015   David Snyder
#           2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#           2015   Johns Hopkins University (Author: Daniel Povey)
#           2017   Radboud University (Author Emre Yilmaz)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
famecorpus=./corpus
num_components=2048

# Data preparation

if [ -d $famecorpus ] ; then
  echo "Fame corpus present. OK."
elif [ -f ./fame.tar.gz ] ; then
  echo "Unpacking..."
  tar xzf fame.tar.gz
elif [ ! -d $famecorpus ] && [ ! -f ./fame.tar.gz ] ; then
  echo "The Fame! corpus is not present. Please register here: http://www.ru.nl/clst/datasets/ "
  echo " and download the corpus and put it at $famecorpus" && exit 1
fi

echo "Preparing data/train.."
local/prepare_train.sh $famecorpus/SC

for task in complete ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do

      echo "Preparing data/fame_${task}_${subtask}_${sets}.."
      trials_female=data/fame_${task}_${subtask}_${sets}_female/trials
      trials_male=data/fame_${task}_${subtask}_${sets}_male/trials
      trials=data/fame_${task}_${subtask}_${sets}/trials
      local/make_fame_test.pl $famecorpus/SV data $task $subtask $sets
      local/make_fame_train.pl $famecorpus/SV data $task $subtask $sets 

    done
  done
done

for task in ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do
      for year in _1t3 _4t10 _mt10; do

        echo "Preparing data/fame_${task}_${subtask}_${sets}${year}.."
        trials_female=data/fame_${task}_${subtask}_${sets}${year}_female/trials
        trials_male=data/fame_${task}_${subtask}_${sets}${year}_male/trials
        trials=data/fame_${task}_${subtask}_${sets}${year}/trials
        local/make_fame_test_year.pl $famecorpus/SV data $task $subtask $sets $year
        local/make_fame_train_year.pl $famecorpus/SV data $task $subtask $sets $year 

      done
    done
  done
done

# MFCC extraction

echo "Extracting MFCC features for data/train.."

steps/make_mfcc.sh --mfcc-config conf/mfcc_16k.conf --nj 100 --cmd "$train_cmd" \
    data/train exp/make_mfcc $mfccdir
utils/fix_data_dir.sh data/train

for task in complete ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do

      echo "Extracting MFCC features for data/fame_${task}_${subtask}_${sets}.."
      steps/make_mfcc.sh --mfcc-config conf/mfcc_16k.conf --nj 100 --cmd "$train_cmd" \
          data/fame_${task}_${subtask}_${sets}_enroll exp/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/fame_${task}_${subtask}_${sets}_enroll
      steps/make_mfcc.sh --mfcc-config conf/mfcc_16k.conf --nj 100 --cmd "$train_cmd" \
          data/fame_${task}_${subtask}_${sets}_test exp/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/fame_${task}_${subtask}_${sets}_test
      
    done
  done
done

for task in ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do
      for year in _1t3 _4t10 _mt10; do

        echo "Extracting MFCC features for data/fame_${task}_${subtask}_${sets}${year}.."
        steps/make_mfcc.sh --mfcc-config conf/mfcc_16k.conf --nj 100 --cmd "$train_cmd" \
            data/fame_${task}_${subtask}_${sets}${year}_enroll exp/make_mfcc $mfccdir
        utils/fix_data_dir.sh data/fame_${task}_${subtask}_${sets}${year}_enroll
        steps/make_mfcc.sh --mfcc-config conf/mfcc_16k.conf --nj 100 --cmd "$train_cmd" \
            data/fame_${task}_${subtask}_${sets}${year}_test exp/make_mfcc $mfccdir
        utils/fix_data_dir.sh data/fame_${task}_${subtask}_${sets}${year}_test

      done
    done
  done
done

# VAD computation

echo "Computing VAD for data/train.."

sid/compute_vad_decision.sh --nj 100 --cmd "$train_cmd" \
    data/train exp/make_vad $vaddir

for task in complete ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do

      echo "Computing VAD for data/fame_${task}_${subtask}_${sets}.."
      sid/compute_vad_decision.sh --nj 100 --cmd "$train_cmd" \
          data/fame_${task}_${subtask}_${sets}_enroll exp/make_vad $vaddir
      sid/compute_vad_decision.sh --nj 100 --cmd "$train_cmd" \
          data/fame_${task}_${subtask}_${sets}_test exp/make_vad $vaddir 

    done
  done
done

for task in ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do
      for year in _1t3 _4t10 _mt10; do

        echo "Computing VAD for data/fame_${task}_${subtask}_${sets}${year}.."
        sid/compute_vad_decision.sh --nj 100 --cmd "$train_cmd" \
            data/fame_${task}_${subtask}_${sets}${year}_enroll exp/make_vad $vaddir
        sid/compute_vad_decision.sh --nj 100 --cmd "$train_cmd" \
            data/fame_${task}_${subtask}_${sets}${year}_test exp/make_vad $vaddir
      
      done
    done
  done
done


# Train UBM and i-vector extractor

echo "Training UBM and the i-vector extractor.."

sid/train_diag_ubm.sh --nj 40 --cmd "$train_cmd" \
    data/train $num_components \
    exp/diag_ubm_$num_components

sid/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
    --cmd "$train_cmd" data/train \
    exp/diag_ubm_$num_components exp/full_ubm_$num_components

sid/train_ivector_extractor.sh --cmd "$train_cmd" --nj 20 --num-threads 4 --num-processes 2 \
  --ivector-dim 600 \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train \
  exp/extractor

# Extract i-vectors

echo "Extracting i-vectors for data/train.."

sid/extract_ivectors.sh --cmd "$train_cmd" --nj 100 \
   exp/extractor data/train \
   exp/ivectors_train 

for task in complete ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do

      echo "Extracting i-vectors for data/fame_${task}_${subtask}_${sets}"
      sid/extract_ivectors.sh --cmd "$train_cmd" --nj 100 \
         exp/extractor data/fame_${task}_${subtask}_${sets}_enroll \
         exp/ivectors_fame_${task}_${subtask}_${sets}_enroll
      sid/extract_ivectors.sh --cmd "$train_cmd" --nj 100 \
         exp/extractor data/fame_${task}_${subtask}_${sets}_test \
         exp/ivectors_fame_${task}_${subtask}_${sets}_test

    done
  done
done

for task in ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do
      for year in _1t3 _4t10 _mt10; do

        echo "Extracting i-vectors for data/fame_${task}_${subtask}_${sets}${year}"
        sid/extract_ivectors.sh --cmd "$train_cmd" --nj 100 \
           exp/extractor data/fame_${task}_${subtask}_${sets}${year}_enroll \
           exp/ivectors_fame_${task}_${subtask}_${sets}${year}_enroll
        sid/extract_ivectors.sh --cmd "$train_cmd" --nj 100 \
           exp/extractor data/fame_${task}_${subtask}_${sets}${year}_test \
           exp/ivectors_fame_${task}_${subtask}_${sets}${year}_test 

      done  
    done
  done
done
 
# Calculate i-vector means used by the scoring scripts

echo "Calculating i-vectors means.."

for task in complete ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do

      local/scoring_common.sh data/train data/fame_${task}_${subtask}_${sets}_enroll data/fame_${task}_${subtask}_${sets}_test \
        exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}_enroll exp/ivectors_fame_${task}_${subtask}_${sets}_test

      trials_female=data/fame_${task}_${subtask}_${sets}_test_female/trials
      trials_male=data/fame_${task}_${subtask}_${sets}_test_male/trials
      trials=data/fame_${task}_${subtask}_${sets}_test/trials

      local/plda_scoring.sh data/train data/fame_${task}_${subtask}_${sets}_enroll data/fame_${task}_${subtask}_${sets}_test \
        exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}_enroll exp/ivectors_fame_${task}_${subtask}_${sets}_test $trials local/scores_gmm_2048_ind_pooled_${task}_${subtask}_${sets}

      local/plda_scoring.sh --use-existing-models true data/train data/fame_${task}_${subtask}_${sets}_enroll_female data/fame_${task}_${subtask}_${sets}_test_female \
        exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}_enroll_female exp/ivectors_fame_${task}_${subtask}_${sets}_test_female $trials_female local/scores_gmm_2048_ind_female_${task}_${subtask}_${sets}

      local/plda_scoring.sh --use-existing-models true data/train data/fame_${task}_${subtask}_${sets}_enroll_male data/fame_${task}_${subtask}_${sets}_test_male \
        exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}_enroll_male exp/ivectors_fame_${task}_${subtask}_${sets}_test_male $trials_male local/scores_gmm_2048_ind_male_${task}_${subtask}_${sets}
              
    done
  done
done

for task in ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do
      for year in _1t3 _4t10 _mt10; do

        local/scoring_common.sh data/train data/fame_${task}_${subtask}_${sets}${year}_enroll data/fame_${task}_${subtask}_${sets}${year}_test \
          exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}${year}_enroll exp/ivectors_fame_${task}_${subtask}_${sets}${year}_test

        trials_female=data/fame_${task}_${subtask}_${sets}${year}_test_female/trials
        trials_male=data/fame_${task}_${subtask}_${sets}${year}_test_male/trials
        trials=data/fame_${task}_${subtask}_${sets}${year}_test/trials

        local/plda_scoring.sh data/train data/fame_${task}_${subtask}_${sets}${year}_enroll data/fame_${task}_${subtask}_${sets}${year}_test \
          exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}${year}_enroll exp/ivectors_fame_${task}_${subtask}_${sets}${year}_test $trials local/scores_gmm_2048_ind_pooled_${task}_${subtask}_${sets}${year}

        local/plda_scoring.sh --use-existing-models true data/train data/fame_${task}_${subtask}_${sets}${year}_enroll_female data/fame_${task}_${subtask}_${sets}${year}_test_female \
          exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}${year}_enroll_female exp/ivectors_fame_${task}_${subtask}_${sets}${year}_test_female $trials_female local/scores_gmm_2048_ind_female_${task}_${subtask}_${sets}${year}

        local/plda_scoring.sh --use-existing-models true data/train data/fame_${task}_${subtask}_${sets}${year}_enroll_male data/fame_${task}_${subtask}_${sets}${year}_test_male \
          exp/ivectors_train exp/ivectors_fame_${task}_${subtask}_${sets}${year}_enroll_male exp/ivectors_fame_${task}_${subtask}_${sets}${year}_test_male $trials_male local/scores_gmm_2048_ind_male_${task}_${subtask}_${sets}${year}

      done
    done
  done
done

# Calculating EER 

echo "Calculating EER.."

for task in complete ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do

      trials=data/fame_${task}_${subtask}_${sets}_test/trials
      echo "GMM-$num_components EER for fame_${task}_${subtask}_${sets}"
      for x in ind; do
        for y in female male pooled; do
          echo "python local/prepare_for_eer.py $trials local/scores_gmm_${num_components}_${x}_${y}_${task}_${subtask}_${sets}/plda_scores"
          eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_gmm_${num_components}_${x}_${y}_${task}_${subtask}_${sets}/plda_scores) 2> /dev/null`
          echo "${x} ${y}: $eer"
        done
      done

    done
  done
done

for task in ageing; do
  for subtask in 3sec 10sec 30sec; do
    for sets in eval; do
      for year in _1t3 _4t10 _mt10; do

        trials=data/fame_${task}_${subtask}_${sets}${year}_test/trials
        echo "GMM-$num_components EER for fame_${task}_${subtask}_${sets}${year}"
        for x in ind; do
          for y in female male pooled; do
            echo "python local/prepare_for_eer.py $trials local/scores_gmm_${num_components}_${x}_${y}_${task}_${subtask}_${sets}${year}/plda_scores"
            eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_gmm_${num_components}_${x}_${y}_${task}_${subtask}_${sets}${year}/plda_scores) 2> /dev/null`
            echo "${x} ${y}: $eer"
          done
        done

      done
    done
  done
done
