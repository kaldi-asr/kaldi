#!/usr/bin/env bash
#
# JHU-CLSP submission for Chime-6 Track 2.
# This recipe uses the ASR from the s5b_track1 recipe. We also replace the 
# baseline SAD with one that uses posterior fusion from the array outputs, 
# and the baseline AHC diarization with VB-based overlap assignment.
#
# This is essentially a decoding script using pretrained models. For training
# scripts for SAD and diarization, please refer to s5_track2 recipe.
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
#            2019  Desh Raj, David Snyder, Ashish Arora
# Apache 2.0

# Begin configuration section.
nj=8
stage=-1
sad_stage=0
score_sad=true
diarizer_stage=0
decode_diarize_stage=0
score_stage=0

enhancement=beamformit  # we will use GSS after diarization

# GSS config (multi-array GSS)
context_samples=320000
iterations=5
ref_array_gss=U01

# RNNLM rescore options
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true
rnnlm_dir=exp/rnnlm_lstm_1b

asr_model_dir=exp/chain_train_worn_simu_u400k_cleaned_rvb/

# option to use the new RTTM reference for sad and diarization
use_new_rttm_reference=false
if $use_new_rttm_reference == "true"; then
  git clone https://github.com/nateanl/chime6_rttm
fi

# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora5/CHiME5
# chime6 data directories, which are generated from ${chime5_corpus},
# to synchronize audio files across arrays and modify the annotation (JSON) file accordingly
chime6_corpus=${PWD}/CHiME6
json_dir=${chime6_corpus}/transcriptions
audio_dir=${chime6_corpus}/audio

enhanced_dir=enhanced
enhanced_dir=$(utils/make_absolute.sh $enhanced_dir) || exit 1

# training data
train_set=train_worn_u400k
test_sets="dev_${enhancement}_dereverb eval_${enhancement}_dereverb"

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e # exit on error

# This script also needs the phonetisaurus g2p, srilm, beamformit
# ./local/check_tools.sh || exit 1;

# This stage demonstrates training of the i-vector extractor required
# for VB resegmentation. The pretrained extractor can be downloaded
# along with all other models in the next stage.
if [ $stage -le -2 ]; then
  if [ ! -d $chime6_corpus ]; then
    echo "$0: Please run stage 0 to prepare CHiME-6 wav files (with WPE)" && exit 1
  fi

  echo "$0: Training UBM and i-vector extractor for VB resegmentation"
  
  # First prepare training data
  for mictype in worn u01 u02 u05 u06; do
    local/prepare_data.sh --mictype ${mictype} --train true \
        ${audio_dir}/train ${json_dir}/train data/train_${mictype}
  done
  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
  utils/copy_data_dir.sh data/train_worn data/train_worn_org # back up
  grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_worn_org/text > data/train_worn/text
  utils/fix_data_dir.sh data/train_worn

  # Remove S12_U05 from training data since it has known issues
  utils/copy_data_dir.sh data/train_u05 data/train_u05_org # back up
  grep -v -e "^S12_U05" data/train_u05_org/text > data/train_u05/text
  utils/fix_data_dir.sh data/train_u05

  # combine mix array and worn mics
  # randomly extract first 400k utterances from all mics
  # if you want to include more training data, you can increase the number of array mic utterances
  utils/combine_data.sh data/train_uall data/train_u01 data/train_u02 data/train_u05 data/train_u06
  utils/subset_data_dir.sh data/train_uall 400000 data/train_u400k
  utils/combine_data.sh data/${train_set} data/train_worn data/train_u400k

  echo "$0:  make features..."
  mfccdir=mfcc
  steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" \
             --mfcc-config conf/mfcc_hires.conf \
             data/${train_set} exp/make_mfcc/${train_set} $mfccdir
  steps/compute_cmvn_stats.sh data/${train_set} exp/make_mfcc/${train_set} $mfccdir
  utils/fix_data_dir.sh data/${train_set}
  
  sid/compute_vad_decision.sh --nj 32 --cmd "$train_cmd" \
    data/${train_set} exp/make_vad
  utils/fix_data_dir.sh data/${train_set}
  
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 10G" --nj 32 \
    --num-threads 8 --subsample 1 --delta-order 0 --apply-cmn false \
    data/${train_set} 1024 exp/vb_reseg/diag_ubm_1024

  diarization/train_ivector_extractor_diag.sh --cmd "$train_cmd --mem 10G" \
    --ivector-dim 400 --num-iters 5 --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 32 \
    exp/vb_reseg/diag_ubm_1024/final.dubm data/${train_set} \
    exp/vb_reseg/extractor_diag_c1024_i400
fi

# This stage downloads all pretrained models required for the inference
# pipeline below.
if [ $stage -le -1 ]; then
  echo "$0: Downloading and extracting pretrained models for inference"
  # Download pretrained SAD if not present
  if [ ! -d exp/segmentation_1a ]; then
    echo "$0: Downloading CHiME-6 baseline SAD"
    wget -O 0012_sad_v1.tar.gz http://kaldi-asr.org/models/12/0012_sad_v1.tar.gz
    tar -xvzf 0012_sad_v1.tar.gz
    cp -r 0012_sad_v1/exp/segmentation_1a exp/
  fi

  # Download i-vector and x-vector extractor if not present
  if [ ! -d exp/xvector_nnet_1a ] || [ ! -d exp/vb_reseg/extractor_diag_c1024_i400 ]; then
    echo "$0: Downloading JHU-CLSP CHiME-6 i-vector and x-vector extractors"
    wget -O 0012_diarization_v2.tar.gz http://kaldi-asr.org/models/12/0012_diarization_v2.tar.gz
    tar -xvzf 0012_diarization_v2.tar.gz
    cp -r 0012_diarization_v2/exp/* exp/
  fi

  # Download acoustic and language models if not present
  if [ ! -d exp/rnnlm_lstm_1b ]; then
    echo "$0: Downloading JHU-CLSP CNN-TDNNF acoustic model and RNNLM"
    wget -O 0012_asr_v2.tar.gz http://kaldi-asr.org/models/12/0012_asr_v2.tar.gz
    tar -xvzf 0012_asr_v2.tar.gz
    cp -r 0012_asr_v2/exp/* exp/
  fi
  
fi

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

#######################################################################
# Prepare the dev and eval data with dereverberation (WPE) and
# beamforming.
#######################################################################
if [ $stage -le 1 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  dereverb_dir=${PWD}/wav/wpe/

  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u06; do
      local/run_wpe.sh --nj 4 --cmd "$train_cmd --mem 20G" \
            ${audio_dir}/${dset} \
            ${dereverb_dir}/${dset} \
            ${mictype}
    done
  done

  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
        ${dereverb_dir}/${dset} \
        ${enhanced_dir}/${dset}_${enhancement}_${mictype} \
        ${mictype}
    done
  done

  # Note that for the evaluation sets, we use the flag
  # "--train false". This keeps the files segments, text,
  # and utt2spk with .bak extensions, so that they can
  # be used later for scoring if needed but are not used
  # in the intermediate stages.
  for dset in dev eval; do
    local/prepare_data.sh --mictype ref --train false \
      "${enhanced_dir}/${dset}_${enhancement}_u0*" \
      ${json_dir}/${dset} data/${dset}_${enhancement}_dereverb
  done

fi

if [ $stage -le 2 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
      --mfcc-config conf/mfcc_hires.conf \
      data/$x exp/make_mfcc/$x $mfccdir
  done
fi

#######################################################################
# Perform SAD on the dev/eval data
#######################################################################
nnet_type=stats
dir=exp/segmentation_1a
sad_work_dir=exp/sad_1a_${nnet_type}/
sad_nnet_dir=$dir/tdnn_${nnet_type}_sad_1a

if [ $stage -le 3 ]; then
  for datadir in ${test_sets}; do
    test_set=data/${datadir}
    if [ ! -f ${test_set}/wav.scp ]; then
      echo "$0: Not performing SAD on ${test_set}"
      exit 0
    fi
    # Perform segmentation
    local/segmentation/detect_speech_activity.sh --nj $nj --stage $sad_stage \
      --cmd "$decode_cmd" \
      $test_set $sad_nnet_dir mfcc $sad_work_dir \
      data/${datadir} || exit 1

    test_dir=data/${datadir}_max_seg
    # Generate RTTM file from segmentation performed by SAD. This can
    # be used to evaluate the performance of the SAD as an intermediate
    # step.
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
      ${test_dir}/utt2spk ${test_dir}/segments ${test_dir}/rttm

    if [ $score_sad == "true" ]; then
      echo "Scoring $datadir.."
      # We first generate the reference RTTM from the backed up utt2spk and segments
      # files.
      ref_rttm=${test_set}/ref_rttm
      steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${test_set}/utt2spk.bak \
        ${test_set}/segments.bak ${test_set}/ref_rttm

      # To score, we select just U06 segments from the hypothesis RTTM.
      hyp_rttm=${test_dir}/rttm
      
      if $use_new_rttm_reference == "true"; then
        echo "Use the new RTTM reference."
        mode="$(cut -d'_' -f1 <<<"$datadir")"
        ref_rttm=./chime6_rttm/${mode}_rttm
      fi

      sed 's/_U0[1-6].ENH//g' $ref_rttm | sort -u > $ref_rttm.scoring
      sed 's/.ENH//g' $hyp_rttm > $hyp_rttm.scoring
      cat ./local/uem_file | grep 'U06' | sed 's/_U0[1-6]//g' > ./local/uem_file.tmp
      md-eval.pl -1 -c 0.25 -u ./local/uem_file.tmp -r $ref_rttm.scoring -s $hyp_rttm.scoring |\
        awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
    fi
  done
fi

#######################################################################
# Perform diarization on the dev/eval data
#######################################################################
if [ $stage -le 4 ]; then
  for datadir in ${test_sets}; do
    # First we prepare the SAD output dir for diarization. For this,
    # we need to have the segment file contain all arrays present in
    # wav.scp of the original test directory.
    test_dir=${datadir}_max_seg
    > data/${test_dir}/segments_all
    > data/${test_dir}/utt2spk_all
    cp data/${datadir}/{wav,feats}.scp data/${test_dir}/
    sessions=$(cut -d' ' -f2 data/${test_dir}/segments | sort -u)
    for session in $sessions; do
      echo "$session"
      arrays=$(cut -d' ' -f1 data/${test_dir}/wav.scp | grep "${session:0:3}")
      for array in $arrays; do
        echo "$array"
        grep "$session" data/${test_dir}/segments |\
          sed "s/$session/$array/g" >> data/${test_dir}/segments_all
      done
    done
    mv data/${test_dir}/segments data/${test_dir}/segments.session
    mv data/${test_dir}/segments_all data/${test_dir}/segments

    # Also prepare utt2spk and spk2utt files
    mv data/${test_dir}/utt2spk data/${test_dir}/utt2spk.session
    paste -d' ' <(cut -d' ' -f1 data/${test_dir}/segments ) \
      <(cut -d'-' -f1 data/${test_dir}/segments ) > data/${test_dir}/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${test_dir}/utt2spk > data/${test_dir}/spk2utt

    # Prepare new feats.scp
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
      --mfcc-config conf/mfcc_hires.conf \
      data/${test_dir} exp/make_mfcc/${test_dir} $mfccdir
  done
fi

if [ $stage -le 5 ]; then
  for datadir in ${test_sets}; do
    if $use_new_rttm_reference == "true"; then
      mode="$(cut -d'_' -f1 <<<"$datadir")"
      ref_rttm=./chime6_rttm/${mode}_rttm
    else
      ref_rttm=data/${datadir}/ref_rttm
    fi
    local/diarize_vb.sh --nj $nj --cmd "$decode_cmd" --stage $diarizer_stage \
      --ref-rttm $ref_rttm \
      exp/xvector_nnet_1a \
      data/${datadir}_max_seg \
      exp/${datadir}_max_seg_diarization
  done
fi

#######################################################################
# Perform GSS using diarized output segments
#######################################################################
if [ $stage -le 6 ]; then
  if [ ! -d pb_chime5/ ]; then
    local/install_pb_chime6.sh
  fi

  if [ ! -d pb_chime5/cache/CHiME6/transcriptions/dev ]; then
    (
    cd pb_chime5
    miniconda_dir=$HOME/miniconda3/
    export PATH=$miniconda_dir/bin:$PATH
    make cache/CHiME6 CHIME5_DIR=${chime5_corpus} CHIME6_DIR=${chime6_corpus}
    )
  fi

  for datadir in ${test_sets}; do
    # First we prepare the RTTM for GSS. For this we remove the introductions using UEM,
    # and also remove any segments shorter than a specified length (for GSS).
    # We also convert the RTTM to JSON format.
    out_dir=exp/${datadir}_max_seg_vb
    test_dir=${datadir}_max_seg.bak # contains all arrays
    data_name=$(echo ${datadir} | cut -d'_' -f1)
    sessions=$(cut -d' ' -f2 data/${test_dir}/segments | cut -d'_' -f1 | sort -u)
    > ${out_dir}/rttm 
    for session in $sessions; do
      echo ${session}
      grep "$session" ${out_dir}/VB_rttm_ol |\
        sed "s/.ENH//g" |\
        local/truncate_rttm.py --min-segment-length 0.2 \
          - local/uem_file - |\
        sed "s/U06/${ref_array_gss}/g" |\
        awk '($8=="5"){$8="1"}{print $0}' |\
        local/convert_rttm_to_json.py - pb_chime5/cache/CHiME6/transcriptions/${data_name}/${session}.json
    done
  done

  pushd pb_chime5
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
  $HOME/miniconda3/bin/python -m pb_chime5.database.chime5.create_json -j cache/chime6.json -db cache/CHiME6 --transcription-path cache/CHiME6/transcriptions --chime6
  popd
fi

if [ $stage -le 7 ]; then
  gss_enhanced_dir=${enhanced_dir}/gss_cs${context_samples}_it${iterations}
  mkdir -p ${enhanced_dir}
  for dset in dev eval; do
    echo "$0: Performing GSS-based ennhancement on ${dset}"
    local/run_gss.sh \
      --cmd "$train_cmd --max-jobs-run 80" --nj 100 \
      --bss_iterations $iterations \
      --context_samples $context_samples \
      ${dset} \
      ${gss_enhanced_dir} \
      ${gss_enhanced_dir} || exit 1
  done
fi

if [ $stage -le 8 ]; then
  echo "$0: Renaming GSS ouput wav files for decoding"
  gss_enhanced_dir=${enhanced_dir}/gss_cs${context_samples}_it${iterations}
  for datadir in ${test_sets}; do
    test_dir=${datadir}_max_seg.bak # contains all arrays
    data_name=$(echo ${datadir} | cut -d'_' -f1)
    sessions=$(cut -d' ' -f2 data/${test_dir}/segments | cut -d'_' -f1 | sort -u)
    for session in $sessions; do
      echo "$session"
      for spk in `seq 1 4`; do
        find ${gss_enhanced_dir}/audio/${data_name} -name *.wav -exec \
          rename "s/${spk}_${session}/${session}_U06.ENH-${spk}/" {} \;
      done
    done
  done
fi

if [ $stage -le 9 ]; then
  echo "$0: Preparing data for ASR"
  gss_enhanced_dir=${enhanced_dir}/gss_cs${context_samples}_it${iterations}
  for dset in dev eval; do
    datadir=${dset}_beamformit_dereverb
    local/prepare_data_gss.sh ${gss_enhanced_dir}/audio/${dset} \
      data/${dset}_diarized || exit 1
    cp data/${datadir}/text.bak data/${dset}_diarized/text
  done
fi

if [ $stage -le 10 ]; then
  for dset in dev eval; do
    data_dir=data/${dset}_diarized
    local/nnet3/decode.sh --affix 2stage --acwt 1.0 --post-decode-acwt 10.0 \
      --frames-per-chunk 150 --nj $nj --ivector-dir exp/nnet3_train_worn_simu_u400k_cleaned_rvb \
      $data_dir data/lang $asr_model_dir/tree_sp/graph $asr_model_dir/tdnn1b_cnn_sp/
  done
fi

if [ $stage -le 11 ]; then
  for dset in dev eval; do
    echo "$0: Perform RNNLM lattice-rescoring on $dset"
    decode_dir=${asr_model_dir}/tdnn1b_cnn_sp/decode_${dset}_diarized_2stage
    # Lattice rescoring
    rnnlm/lmrescore_pruned.sh \
        --cmd "$decode_cmd --mem 8G" --skip-scoring true \
        --weight 0.45 --max-ngram-order $ngram_order \
        data/lang $rnnlm_dir \
        data/${dset}_diarized ${decode_dir} \
        ${asr_model_dir}/decode_${dset}_diarized_2stage_rescore
    local/score.sh --cmd "$decode_cmd --mem 8G" data/${dset}_diarized data/lang \
      ${asr_model_dir}/decode_${dset}_diarized_2stage_rescore
  done
fi

#######################################################################
# Score decoded dev/eval sets
#######################################################################
if [ $stage -le 12 ]; then
  # final scoring to get the challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh --stage $score_stage \
      --dev_decodedir ${asr_model_dir}/tdnn1b_cnn_sp/decode_dev_diarized_2stage_rescore \
      --dev_datadir dev_diarized \
      --eval_decodedir ${asr_model_dir}/tdnn1b_cnn_sp/decode_eval_diarized_2stage_rescore \
      --eval_datadir eval_diarized
fi
exit 0;
