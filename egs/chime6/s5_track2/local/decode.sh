#!/bin/bash
#
# This script decodes raw utterances through the entire pipeline: 
# Feature extraction -> SAD -> Diarization -> ASR
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
#            2019  Desh Raj, David Snyder, Ashish Arora
# Apache 2.0

# Begin configuration section.
nj=50
decode_nj=10
stage=0
sad_stage=0
diarizer_stage=0
enhancement=
test_sets=
skip_scoring=false
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh
. ./conf/sad.conf

#######################################################################
# Prepare the dev and eval data with dereverberation (WPE) and
# beamforming.
#######################################################################
if [ $stage -le 0 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhandir=enhan
  dereverb_dir=${PWD}/wav/wpe/

  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u06; do
      local/run_wpe.sh --nj 4 --cmd "$train_cmd --mem 120G" \
            ${audio_dir}/${dset} \
            ${dereverb_dir}/${dset} \
            ${mictype}
    done
  done

  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
			      ${dereverb_dir}/${dset} \
			      ${enhandir}/${dset}_${enhancement}_${mictype} \
			      ${mictype}
    done
  done

  for dset in dev eval; do
    local/prepare_data.sh --mictype ref --train false \
      "$PWD/${enhandir}/${dset}_${enhancement}_u0*" \
			${json_dir}/${dset} data/${dset}_${enhancement}_dereverb_ref 
  done
fi

if [ $stage -le 1 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${test_sets}; do
    steps/make_mfcc.sh --nj $decode_nj --cmd "$train_cmd" \
      --mfcc-config conf/mfcc_hires.conf \
		  data/$x exp/make_mfcc/$x $mfccdir
  done
fi

#######################################################################
# Perform SAD on the dev/eval data
#######################################################################
dir=exp/segmentation${affix}
sad_work_dir=exp/sad${affix}_${nnet_type}/
sad_nnet_dir=$dir/tdnn_${nnet_type}_sad_1a

if [ $stage -le 2 ]; then
  for datadir in ${test_sets}; do
    test_set=data/${datadir}
    if [ ! -f ${test_set}/wav.scp ]; then
      echo "$0: Not performing SAD on ${test_set}"
      exit 0
    fi
    # Perform segmentation
    local/segmentation/detect_speech_activity.sh --nj $decode_nj --stage $sad_stage \
      $test_set $sad_nnet_dir mfcc $sad_work_dir \
      data/${datadir} || exit 1
   
    mv data/${datadir}_seg data/${datadir}_${nnet_type}_seg
    # Generate RTTM file from segmentation performed by SAD. This can
    # be used to evaluate the performance of the SAD as an intermediate
    # step.
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
      data/${datadir}_${nnet_type}_seg/utt2spk data/${datadir}_${nnet_type}_seg/segments \
      data/${datadir}_${nnet_type}_seg/rttm
  done
fi

#######################################################################
# Perform diarization on the dev/eval data
#######################################################################
if [ $stage -le 3 ]; then
  for datadir in ${test_sets}; do
    local/diarize.sh --nj 10 --cmd "$train_cmd" --stage $diarizer_stage \
      exp/xvector_nnet_1a \
      data/${datadir}_${nnet_type}_seg \
      exp/${datadir}_${nnet_type}_seg_diarization
  done
fi

#######################################################################
# Decode diarized output using trained chain model
#######################################################################
if [ $stage -le 4 ]; then
  continue
fi

#######################################################################
# Score decoded dev/eval sets
#######################################################################
if [ $skip_scoring == "false" ]; then
  continue
fi
exit 0;
