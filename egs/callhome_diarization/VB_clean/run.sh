#!/bin/bash

# This example demonstrates how to perform VB resegmentation for callhome dataset.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
num_gauss=1024
ivec_dim=400
label_rttm_file=data/callhome/fullref.rttm
init_rttm_file=rttm/x_vector_rttm
output_dir=exp/xvec_init_gauss_${num_gauss}_ivec_${ivec_dim}

stage=0

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh $data_root data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare the Callhome portion of NIST SRE 2000.
  local/make_callhome.sh /export/corpora/NIST/LDC2001S97/ data/
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre swbd; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  utils/combine_data.sh data/swbd_sre data/swbd data/sre
  utils/subset_data_dir.sh data/swbd_sre 32000 data/swbd_sre_32k

  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
    --cmd "$train_cmd" --write-utt2num-frames true \
    data/callhome exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/callhome
fi

if [ $stage -le 2 ]; then
  # Apply cmn and adding deltas will harm the performance on the callhome dataset. So we just use the 20-dim raw MFCC feature.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G --max-jobs-run 6" \
    --nj 20 --num-threads 4  --subsample 1 --delta-order 0 --apply-cmn false \
    data/swbd_sre_32k $num_gauss \
    exp/diag_ubm_gauss_${num_gauss}_delta_0_cmn_0
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor. The UBM is assumed to be diagonal.
  local/train_ivector_extractor_diag.sh --cmd "$train_cmd --mem 45G --max-jobs-run 20" \
    --ivector-dim ${ivec_dim} \
    --num-iters 5 \
    --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 10 \
    exp/diag_ubm_gauss_${num_gauss}_delta_0_cmn_0/final.dubm data/swbd_sre \
    exp/extractor_gauss_${num_gauss}_delta_0_cmn_0_ivec_${ivec_dim}
fi

if [ $stage -le 4 ]; then
  # Convert the Kaldi UBM and T-matrix model to numpy array.
  mkdir -p $output_dir
  mkdir -p $output_dir/tmp
  mkdir -p $output_dir/log
  mkdir -p $output_dir/model

  # Dump the diagonal UBM model into text format.
  "$train_cmd" $output_dir/log/convert_diag_ubm.log \
    gmm-global-copy --binary=false \
      exp/diag_ubm_gauss_${num_gauss}_delta_0_cmn_0/final.dubm \
      $output_dir/tmp/dubm.tmp || exit 1;

  # Dump the ivector extractor model into text format.
  # This method is not currently supported by Kaldi, 
  # so please use my kaldi.  
  "$train_cmd" $output_dir/log/convert_ie.log \
    ivector-extractor-copy --binary=false \
      exp/extractor_gauss_${num_gauss}_delta_0_cmn_0_ivec_${ivec_dim}/final.ie \
      $output_dir/tmp/ie.tmp || exit 1;
   
  local/dump_model.py $output_dir/tmp/dubm.tmp $output_dir/model
  local/dump_model.py $output_dir/tmp/ie.tmp $output_dir/model 
fi

if [ $stage -le 5 ]; then
  mkdir -p $output_dir/results

  # Compute the DER before VB resegmentation
  md-eval.pl -1 -c 0.25 -r $label_rttm_file -s $init_rttm_file 2> $output_dir/log/DER_init.log \
    > $output_dir/results/DER_init.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $output_dir/results/DER_init.txt)
  # Before VB resegmentation, DER: 8.02%
  echo "Before VB resegmentation, DER: $der%"

  # VB resegmentation. In this script, I use the x-vector result to 
  # initialize the VB system. You can also use i-vector result or random 
  # initize the VB system.
  local/VB_resegmentation.sh --nj 20 --cmd "$train_cmd --mem 10G" \
    --true_rttm_filename "None" --initialize 1 \
    data/callhome $init_rttm_file $output_dir $output_dir/model/diag_ubm.pkl $output_dir/model/ie.pkl || exit 1; 

  # Compute the DER after VB resegmentation
  cat $output_dir/rttm/* > $output_dir/predict.rttm
  md-eval.pl -1 -c 0.25 -r $label_rttm_file -s $output_dir/predict.rttm 2> $output_dir/log/DER.log \
    > $output_dir/results/DER.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $output_dir/results/DER.txt)
  # After VB resegmentation, DER: 6.15%
  echo "After VB resegmentation, DER: $der%"
fi
