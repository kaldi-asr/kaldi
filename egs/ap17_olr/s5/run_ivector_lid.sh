#!/bin/bash
# Copyright 2013   Daniel Povey
#           2014   David Snyder
#           2016   Lantian Li, Yixiang Chen, Zhiyuan Tang, Dong Wang
# Apache 2.0.
#

# This example script for i-vector LID is still a bit of a mess, and needs to be
# cleaned up, but it shows you all the basic ingredients.

# The default path of training set is 'data/train',
# the default path of test set is 'data/test'.
# and the default path of dev set is 'data/{dev_1s,dev_3s,dev_all}'.


. ./cmd.sh
. ./path.sh

# Number of components
cnum=2048 # num of Gaussions
civ=400   # dim of i-vector
clda=6    # dim for i-vector with LDA

testdir=dev_all  # you may change it to dev_1s, dev_3s or test
exp=exp/ivector
mkdir -p $exp

stage=1

set -e

# When doing i-vector LID, the two files ('spk2utt' and 'utt2spk') in data/{train,dev_1s,dev_3s,dev_all},
# should be replaced with those in local/olr/ivector/lan2utt, 
# after then, each 'spk' actually indicates one language of the ten.
# Those files in data/{train,dev_1s,dev_3s,dev_all}/lan2utt are outdated. 
for subpath in {train,dev_1s,dev_3s,dev_all}/utt2spk; do
  if ! cmp data/$subpath local/olr/ivector/lan2utt/$subpath &> /dev/null; then
    echo "data/$subpath should be replaced with local/olr/ivector/lan2utt/$subpath and so is spk2utt" && exit 1;
  fi
done

if [ $stage -le 1 ];then
  # Feature extraction and VAD
  mfccdir=`pwd`/_mfcc
  vaddir=`pwd`/_vad
  
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 10 --cmd "$cpu_cmd" data/train $exp/_log_mfcc $mfccdir
  sid/compute_vad_decision.sh --nj 4 --cmd "$cpu_cmd" data/train $exp/_log_vad $vaddir
  
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 10 --cmd "$cpu_cmd" data/$testdir $exp/_log_mfcc $mfccdir
  sid/compute_vad_decision.sh --nj 4 --cmd "$cpu_cmd" data/$testdir $exp/_log_vad $vaddir
fi

if [ $stage -le 2 ];then
  # Get smaller subsets of training data for faster training.
  utils/subset_data_dir.sh data/train 18000 data/train_18k
  utils/subset_data_dir.sh data/train 36000 data/train_36k

  # UBM and T-matrix training
  sid/train_diag_ubm.sh --nj 6 --cmd "$cpu_cmd" data/train_18k ${cnum} $exp/diag_ubm_${cnum} 
  sid/train_full_ubm.sh --nj 6 --cmd "$cpu_cmd" data/train_36k $exp/diag_ubm_${cnum} $exp/full_ubm_${cnum} 

  sid/train_ivector_extractor.sh --nj 10 --cmd "$cpu_cmd -l mem_free=2G" \
    --num-iters 6 --ivector_dim $civ $exp/full_ubm_${cnum}/final.ubm data/train \
    $exp/extractor_${cnum}_${civ}

  # Extract i-vector
  sid/extract_ivectors.sh --cmd "$cpu_cmd -l mem_free=2G," --nj 20 \
    $exp/extractor_${cnum}_${civ} data/train $exp/ivectors_train_${cnum}_${civ}
  sid/extract_ivectors.sh --cmd "$cpu_cmd -l mem_free=2G," --nj 20 \
    $exp/extractor_${cnum}_${civ} data/$testdir $exp/ivectors_${testdir}_${cnum}_${civ}
fi

# Demonstrate simple cosine-distance scoring
if [ $stage -le 3 ];then
  trials=local/olr/ivector/trials.trl
  mkdir -p $exp/score/total
  cat $trials | awk '{print $1, $2}' | \
   ivector-compute-dot-products - \
    scp:$exp/ivectors_train_${cnum}_${civ}/spk_ivector.scp \
    scp:$exp/ivectors_${testdir}_${cnum}_${civ}/ivector.scp \
    $exp/score/total/foo_cosine
  
  echo i-vector
  echo
  printf '% 16s' 'EER% is:'
  eer=$(awk '{print $3}' $exp/score/total/foo_cosine | paste - $trials | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  printf '% 5.2f' $eer
  echo
  
  python local/olr/ivector/Compute_Cavg.py  $exp/score/total/foo_cosine data/${testdir}/utt2spk
fi

# Demonstrate what happens if we reduce the dimension with LDA
if [ $stage -le 4 ];then
  ivector-compute-lda --dim=$clda  --total-covariance-factor=0.1 \
    "ark:ivector-normalize-length scp:$exp/ivectors_train_${cnum}_${civ}/ivector.scp  ark:- |" \
    ark:data/train/utt2spk \
    $exp/ivectors_train_${cnum}_${civ}/transform_${clda}.mat
  
  ivector-transform $exp/ivectors_train_${cnum}_${civ}/transform_${clda}.mat \
    scp:$exp/ivectors_train_${cnum}_${civ}/ivector.scp ark:- | \
    ivector-normalize-length ark:- ark:${exp}/ivectors_train_${cnum}_${civ}/lda_${clda}.ark
  
  ivector-transform $exp/ivectors_train_${cnum}_${civ}/transform_${clda}.mat \
    scp:$exp/ivectors_${testdir}_${cnum}_${civ}/ivector.scp ark:- | \
    ivector-normalize-length ark:- ark:${exp}/ivectors_${testdir}_${cnum}_${civ}/lda_${clda}.ark
fi

# Demonstrate cosine-distance scoring for i-vector with LDA
if [ $stage -le 5 ];then
  dir=${exp}/ivectors_train_${cnum}_${civ}
  
  ivector-mean ark:data/train/spk2utt \
    ark:$dir/lda_${clda}.ark ark:- ark,t:$dir/num_utts.ark | \
    ivector-normalize-length ark:- ark,scp:$dir/lda_ivector.ark,$dir/lda_ivector.scp
  
  trials=local/olr/ivector/trials.trl
  cat $trials | awk '{print $1, $2}' | \
   ivector-compute-dot-products - \
    scp:$exp/ivectors_train_${cnum}_${civ}/lda_ivector.scp \
    ark:$exp/ivectors_${testdir}_${cnum}_${civ}/lda_${clda}.ark \
    $exp/score/total/foo_lda
  
  echo 'L-vector (i-vector with LDA)'
  echo 
  printf '% 16s' 'EER% is:'
  eer=$(awk '{print $3}' $exp/score/total/foo_lda | paste - $trials | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  printf '% 5.2f' $eer
  
  python local/olr/ivector/Compute_Cavg.py  $exp/score/total/foo_lda data/${testdir}/utt2spk
fi

# Evaluation with SVM
vectype=lda-ivector # or just i-vector
curve=linear
maxit=-1

mkdir -p svm
mkdir -p exp/ivector_svm/$curve
mkdir -p exp/ivector_svm/score

# using i-vector directly or i-vector with LDA for SVM
if [ $vectype == "lda-ivector" ]; then
  # using i-vector with LDA
  ivector-normalize-length ark:$exp/ivectors_train_${cnum}_${civ}/lda_${clda}.ark ark,t:exp/ivector_svm/$curve/train.dat
  ivector-normalize-length ark:$exp/ivectors_${testdir}_${cnum}_${civ}/lda_${clda}.ark ark,t:exp/ivector_svm/$curve/test.dat
elif [ $vectype == "i-vector" ]; then
  # using i-vecotr directly
  ivector-normalize-length scp:$exp/ivectors_train_${cnum}_${civ}/ivector.scp ark,t:exp/ivector_svm/$curve/train.dat
  ivector-normalize-length scp:$exp/ivectors_${testdir}_${cnum}_${civ}/ivector.scp ark,t:exp/ivector_svm/$curve/test.dat
else
  echo "Wrong feats!" && exit 1;
fi

if [ $stage -le 6 ];then
  python local/olr/ivector/pre_svmd.py exp/ivector_svm/$curve/train.dat data/train/utt2spk exp/ivector_svm/$curve/svmtrain.dat
  python local/olr/ivector/pre_svmd.py exp/ivector_svm/$curve/test.dat data/${testdir}/utt2spk exp/ivector_svm/$curve/svmtest.dat
  
  python local/olr/ivector/svm_ratelimit.py exp/ivector_svm/$curve/svmtrain.dat exp/ivector_svm/$curve/svmtest.dat exp/ivector_svm/score/foo_${curve}_${maxit}.log $maxit $curve
  
  cat exp/ivector_svm/score/foo_${curve}_${maxit}.log | sed 's/]]//' > exp/ivector_svm/score/foo_${curve}_${maxit}.score
  
  #local/score.sh $dpath/trials.trl exp/ivector_svm/foo_linear.score
  
  python local/olr/ivector/ttt.py exp/ivector_svm/score/foo_${curve}_${maxit}.score exp/ivector_svm/score/foo_standard_${curve}_${maxit}.score
  
  
  echo "$vectype + $curve SVM"
  echo
  
  python local/olr/ivector/pre_eerdata.py exp/ivector_svm/$curve/svmtest.dat exp/ivector_svm/score/foo_standard_${curve}_${maxit}.score exp/ivector_svm/$curve/foo_eerdata_${curve}_${maxit}.dat
  
  printf '% 16s' 'EER% is:'
  eer=$(compute-eer --verbose=2 exp/ivector_svm/$curve/foo_eerdata_${curve}_${maxit}.dat 2>/dev/null)
  printf '% 5.2f' $eer
  echo
  
  python local/olr/ivector/Compute_Cavg_svm.py exp/ivector_svm/score/foo_standard_${curve}_${maxit}.score exp/ivector_svm/$curve/svmtest.dat
fi

exit 0;
