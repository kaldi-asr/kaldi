#!/usr/bin/env bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#
# This script only performs recognition experiments with evaluation data
# This script can be run from run.sh or standalone. 
# To run it standalone, you can download a pretrained chain ASR model using:
# wget http://kaldi-asr.org/models/12/0012_asr_v1.tar.gz
# Once it is downloaded, extract using: tar -xvzf 0012_asr_v1.tar.gz
# and copy the contents of the {data/ exp/} directory to your {data/ exp/}

# Begin configuration section.
decode_nj=20
gss_nj=50
stage=0
enhancement=gss        # for a new enhancement method,
                       # change this variable and stage 4

# training data
train_set=train_worn_simu_u400k
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


set -e # exit on error

# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora4/CHiME5
# chime6 data directories, which are generated from ${chime5_corpus},
# to synchronize audio files across arrays and modify the annotation (JSON) file accordingly
chime6_corpus=${PWD}/CHiME6
json_dir=${chime6_corpus}/transcriptions
audio_dir=${chime6_corpus}/audio

enhanced_dir=enhanced
if [[ ${enhancement} == *gss* ]]; then
  enhanced_dir=${enhanced_dir}_multiarray
  enhancement=${enhancement}_multiarray
fi

if [[ ${enhancement} == *beamformit* ]]; then
  enhanced_dir=${enhanced_dir}
  enhancement=${enhancement}
fi

enhanced_dir=$(utils/make_absolute.sh $enhanced_dir) || exit 1
test_sets="dev_${enhancement} eval_${enhancement}"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

###########################################################################
# We first generate the synchronized audio files across arrays and
# corresponding JSON files. Note that this requires sox v14.4.2,
# which is installed via miniconda in ./local/check_tools.sh
###########################################################################

if [ $stage -le 0 ]; then
  local/generate_chime6_data.sh \
    --cmd "$train_cmd" \
    ${chime5_corpus} \
    ${chime6_corpus}
fi

#########################################################################################
# In stage 1, we perform GSS based enhancement or beamformit for the test sets. multiarray = true
#can take around 10hrs for dev and eval set.
#########################################################################################

if [ $stage -le 1 ] && [[ ${enhancement} == *gss* ]]; then
  echo "$0:  enhance data..."
  # Guided Source Separation (GSS) from Paderborn University
  # http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf
  # @Article{PB2018CHiME5,
  #   author    = {Boeddeker, Christoph and Heitkaemper, Jens and Schmalenstroeer, Joerg and Drude, Lukas and Heymann, Jahn and Haeb-Umbach, Reinhold},
  #   title     = {{Front-End Processing for the CHiME-5 Dinner Party Scenario}},
  #   year      = {2018},
  #   booktitle = {CHiME5 Workshop},
  # }

  if [ ! -d pb_chime5/ ]; then
    local/install_pb_chime5.sh
  fi

  if [ ! -f pb_chime5/cache/chime6.json ]; then
    (
    cd pb_chime5
    miniconda_dir=$HOME/miniconda3/
    export PATH=$miniconda_dir/bin:$PATH
    export CHIME6_DIR=$chime6_corpus
    make cache/chime6.json
    )
  fi

  for dset in dev eval; do
    local/run_gss.sh \
      --cmd "$train_cmd --max-jobs-run $gss_nj" --nj 160 \
      ${dset} \
      ${enhanced_dir} \
      ${enhanced_dir} || exit 1
  done

  for dset in dev eval; do
    local/prepare_data.sh --mictype gss ${enhanced_dir}/audio/${dset} \
      ${json_dir}/${dset} data/${dset}_${enhancement} || exit 1
  done
fi

#######################################################################
# Prepare the dev and eval data with dereverberation (WPE) and
# beamforming.
#######################################################################

if [ $stage -le 1 ] && [[ ${enhancement} == *beamformit* ]]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhanced_dir=enhan
  dereverb_dir=${PWD}/wav/wpe/
  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_wpe.sh --nj 4 --cmd "$train_cmd --mem 20G" \
               ${audio_dir}/${dset} \
               ${dereverb_dir}/${dset} \
               ${mictype}
    done
  done

  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
                      ${dereverb_dir}/${dset} \
                      ${enhanced_dir}/${dset}_${enhancement}_${mictype} \
                      ${mictype}
    done
  done

  for dset in dev eval; do
    local/prepare_data.sh --mictype ref "$PWD/${enhanced_dir}/${dset}_${enhancement}_u0*" \
                      ${json_dir}/${dset} data/${dset}_${enhancement}
  done
fi

# In GSS enhancement, we do not have array information in utterance ID
if [ $stage -le 2 ] && [[ ${enhancement} == *gss* ]]; then
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in ${test_sets}; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_orig
  done

  for dset in ${test_sets}; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_orig data/${dset}
  done
fi

if [ $stage -le 2 ] && [[ ${enhancement} == *beamformit* ]]; then
  # fix speaker ID issue (thanks to Dr. Naoyuki Kanda)
  # add array ID to the speaker ID to avoid the use of other array information to meet regulations
  # Before this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01
  # After this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit_fix/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01_U02
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01_U02
  echo "$0: fix data..."
  for dset in ${test_sets}; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    mkdir -p data/${dset}_nosplit_fix
    for f in segments text wav.scp; do
      if [ -f data/${dset}_nosplit/$f ]; then
        cp data/${dset}_nosplit/$f data/${dset}_nosplit_fix
      fi
    done
    awk -F "_" '{print $0 "_" $3}' data/${dset}_nosplit/utt2spk > data/${dset}_nosplit_fix/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${dset}_nosplit_fix/utt2spk > data/${dset}_nosplit_fix/spk2utt
  done

  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in ${test_sets}; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit_fix data/${dset}
  done
fi

##########################################################################
# DECODING: we perform 2 stage decoding.
##########################################################################

nnet3_affix=_${train_set}_cleaned_rvb
lm_suffix=

if [ $stage -le 3 ]; then
  # First the options that are passed through to run_ivector_common.sh
  # (some of which are also used in this script directly).

  # The rest are configs specific to this script.  Most of the parameters
  # are just hardcoded at this level, in the commands below.
  echo "$0: decode data..."
  affix=1b   # affix for the TDNN directory name
  tree_affix=
  tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
  dir=exp/chain${nnet3_affix}/tdnn${affix}_sp

  # training options
  # training chunk-options
  chunk_width=140,100,160
  # we don't need extra left/right context for TDNN systems.
  chunk_left_context=0
  chunk_right_context=0
  
  utils/mkgraph.sh \
      --self-loop-scale 1.0 data/lang${lm_suffix}/ \
      $tree_dir $tree_dir/graph${lm_suffix} || exit 1;

  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      local/nnet3/decode.sh --affix 2stage --pass2-decode-opts "--min-active 1000" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --frames-per-chunk 150 --nj $decode_nj \
        --ivector-dir exp/nnet3${nnet3_affix} \
        data/${data} data/lang${lm_suffix} \
        $tree_dir/graph${lm_suffix} \
        exp/chain${nnet3_affix}/tdnn${affix}_sp
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

##########################################################################
# Scoring: here we obtain wer per session per location and overall WER
##########################################################################

if [ $stage -le 4 ]; then
  # final scoring to get the official challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh --enhancement $enhancement --json $json_dir \
      --dev exp/chain${nnet3_affix}/tdnn1b_sp/decode${lm_suffix}_dev_${enhancement}_2stage \
      --eval exp/chain${nnet3_affix}/tdnn1b_sp/decode${lm_suffix}_eval_${enhancement}_2stage
fi
