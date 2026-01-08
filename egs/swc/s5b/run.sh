#!/bin/bash

# This Kaldi recipe based on all three recordings in the Sheffield Wargame 
# Corpora (SWC) is forked from the Kaldi recipe "s5b" for AMI data.
# 
# The interface for data downloading is NOT provided because that could not 
# be automated for ethical concerns. Please visit the following website for 
# details about the access to corpus data and language model:
#
#	http://mini-vm20.dcs.shef.ac.uk/swc/corpora_data.html
#
# Yulan Liu, 19 Sep 2016	


. ./cmd.sh
. ./path.sh

echo "Starting time:"
date

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=ihm
MODE='SA1'      # Other options: 'SA2', 'AD1', 'AD2'
fastrun=1	# 1: skip unnecessary decoding 
spkadapt=1	# 1: include speaker adaptation
nj=20 		# defalut number of parallel jobs,
stage=0

. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

set -euo pipefail

# Path where SWC gets downloaded (or where locally available):
SWCDIR=$PWD/wav_db # Default, 
case $(hostname -d) in
  minigrid.dcs.shef.ac.uk) SWCDIR=/share/spandh.ami1/usr/yulan/sw/kaldi/dev/kaldi-trunk/egs/swc/swc ;; # UoS
esac


# Check microphone type
if [ "$base_mic" == "ihm" ] ; then		# Individual headset microphone
  mic_ch=ihm
  PROCESSED_DIR=$SWCDIR
elif [ "$base_mic" == "sdm" ] ; then		# Single distant microphone (default: "TBL1-01")
  mic_ch=TBL1-0${nmics}
  PROCESSED_DIR=$SWCDIR
elif [ "$base_mic" == "mdm" ] ; then		# Multiple distant microphone (default: 8 channels from "TBL1" array)
  mic_ch=TBL1
  # check whether beamforming has been executed already
  BEAMFORM_DIR=$SWCDIR/${nmics}bmit_${mic_ch}
  if [ ! -d $BEAMFORM_DIR ] ; then
    ! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; make beamformit;'" && exit 1
    local/swc_beamform.sh --cmd "$train_cmd" --nj 20 $nmics ${mic_ch} $SWCDIR ${BEAMFORM_DIR}
  fi	  
  PROCESSED_DIR=${BEAMFORM_DIR}
fi


# Prepare original data directories data/ihm/train_orig, etc.
# The files needed for scoring is also prepared here.
if [ $stage -le 2 ]; then
  if [ "$base_mic" == "sdm" ] ; then
    local/swc_${base_mic}_data_prep.sh $PROCESSED_DIR  $MODE  $nmics  --CH ${mic_ch}
  elif [ "$base_mic" == "mdm" ] ; then
    local/swc_${base_mic}_data_prep.sh $SWCDIR  $BEAMFORM_DIR  $MODE  $nmics
  else	  
    local/swc_${base_mic}_data_prep.sh $PROCESSED_DIR  $MODE
  fi
fi


# LM checking
[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
# LM=$final_lm
LM=${final_lm}.pr1-7


# Here starts the normal recipe, which is mostly shared across mic scenarios,
# - for ihm we adapt to speaker by fMLLR,
# - for sdm and mdm we do not adapt for speaker, but for environment only (cmn),


# This function is added compared to SWC recipe version "s5" for better i-vector estimation.
if [ $stage -le 3 ]; then
  for dset in train dev eval; do
    if [ -d data/$mic/$MODE/$dset ]; then  
      mv data/$mic/$MODE/$dset  data/$mic/$MODE/${dset}_orig
      # this splits up the speakers (which for sdm and mdm just correspond
      # to recordings) into 30-second chunks.  It's like a very brain-dead form
      # of diarization; we can later replace it with 'real' diarization.
      seconds_per_spk_max=30
      [ "$mic" == "ihm" ] && seconds_per_spk_max=120  # speaker info for ihm is real,
                                                    # so organize into much bigger chunks.
      utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 \
        data/$mic/$MODE/${dset}_orig data/$mic/$MODE/$dset
    fi
  done
fi


# Feature extraction,
if [ $stage -le 4 ]; then
  for dset in train dev eval; do
    if [ -d data/$mic/$MODE/$dset ]; then
      fd=data/$mic/$MODE/$dset
      steps/make_mfcc.sh --nj 15 --cmd "$train_cmd"  $fd  $fd/log  $fd/data
      steps/compute_cmvn_stats.sh  $fd  $fd/log  $fd/data
      utils/fix_data_dir.sh $fd
    fi
  done
fi


# Train systems, adjust the number of parallel jobs to be the number of speakers
nj_dev=$(cat data/$mic/$MODE/dev/spk2utt | wc -l)
nj_eval=$(cat data/$mic/$MODE/eval/spk2utt | wc -l)
fd=data/$mic/$MODE

# monophone training
if [ $stage -le 5 ]; then
  # using all data without sub-sampling since SWC is not that large
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    $fd/train data/lang exp/$mic/$MODE/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $fd/train data/lang exp/$mic/$MODE/mono exp/$mic/$MODE/mono_ali
fi


# context-dep. training with delta features.
if [ $stage -le 6 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000  $fd/train data/lang exp/$mic/$MODE/mono_ali exp/$mic/$MODE/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $fd/train data/lang exp/$mic/$MODE/tri1 exp/$mic/$MODE/tri1_ali
fi


if [ $stage -le 7 ]; then
  # LDA_MLLT
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000  $fd/train data/lang exp/$mic/$MODE/tri1_ali exp/$mic/$MODE/tri2
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $fd/train data/lang exp/$mic/$MODE/tri2 exp/$mic/$MODE/tri2_ali

  if [ $fastrun -ne 1 ]; then
    # Decode
    graph_dir=exp/$mic/$MODE/tri2/graph_${LM}
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${LM} exp/$mic/$MODE/tri2 $graph_dir
    steps/decode.sh --nj ${nj_dev} --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir $fd/dev exp/$mic/$MODE/tri2/decode_dev_${LM}
    steps/decode.sh --nj ${nj_eval} --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir $fd/eval exp/$mic/$MODE/tri2/decode_eval_${LM}
  fi
fi



if [ $stage -le 8 ]; then
  # Train tri3, which is LDA+MLLT,
  echo "LDA+MLLT training - folder \"tri3\". "
  fd=data/$mic/$MODE

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000  $fd/train  data/lang  exp/$mic/$MODE/tri2_ali  exp/$mic/$MODE/tri3
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $fd/train  data/lang  exp/$mic/$MODE/tri3  exp/$mic/$MODE/tri3_ali
  

  # Decode,
  echo "Decoding."
  graph_dir=exp/$mic/$MODE/tri3/graph_${LM}
  $highmem_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh  data/lang_${LM}  exp/$mic/$MODE/tri3  $graph_dir
  steps/decode.sh --nj $nj_dev --cmd "$decode_cmd" --config conf/decode.conf \
    --skip_scoring true	\
    $graph_dir  $fd/dev  exp/$mic/$MODE/tri3/decode_dev_${LM}
  steps/decode.sh --nj $nj_eval --cmd "$decode_cmd" --config conf/decode.conf \
    --skip_scoring true \
    $graph_dir  $fd/eval  exp/$mic/$MODE/tri3/decode_eval_${LM}
fi


nj_mmi=22	# 22 speakers; default: 80
if [ $stage -le 9 ]; then
  if [ $spkadapt -eq 1 ]; then            # With speaker adaptation 
    # LDA+MLLT+SAT
    echo "SAT training - folder \"tri4a\". "
    fd=data/$mic/$MODE


    steps/train_sat.sh  --cmd "$train_cmd" \
      5000 80000  $fd/train  data/lang  exp/$mic/$MODE/tri3_ali  exp/$mic/$MODE/tri4a

    # Decode,  
    echo "Decoding SAT system."
    graph_dir=exp/$mic/$MODE/tri4a/graph_${LM}
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh  data/lang_${LM}  exp/$mic/$MODE/tri4a  $graph_dir
    steps/decode_fmllr.sh --nj $nj_dev --cmd "$decode_cmd"  --config conf/decode.conf \
      --skip_scoring true \
      $graph_dir  $fd/dev  exp/$mic/$MODE/tri4a/decode_dev_${LM}
    steps/decode_fmllr.sh --nj $nj_eval --cmd "$decode_cmd" --config conf/decode.conf \
      --skip_scoring true \
      $graph_dir  $fd/eval  exp/$mic/$MODE/tri4a/decode_eval_${LM}

    steps/align_fmllr.sh --nj $nj_mmi --cmd "$train_cmd" \
      $fd/train  data/lang  exp/$mic/$MODE/tri4a  exp/$mic/$MODE/tri4a_ali
  fi
fi 



# DNN training. This script is based on egs/ami/s5/local/run_dnn.sh
# Some of them would be out of date.
if [ $stage -le 10 ]; then
  if [ $spkadapt -eq 1 ]; then		# With SAT
    echo "Train feed-forward DNN with speaker adaptation based on HMM-GMM system in folder \"tri4a\". "
    local/nnet/run_dnn.sh $mic  $MODE  
  else					# Without SAT
    echo "Train feed-forward DNN without speaker adaptation based on HMM-GMM system in folder \"tri3\". "
    local/nnet/run_dnn_lda_mllt.sh $mic  $MODE
  fi
fi


echo "Baseline system construction finished."
exit;



# TDNN training. As an advanced system by default it is not constructed.
if [ $stage -le 11 ]; then
  echo "Advanced system: TDNN - prepare data."

  if [ $mic == "ihm" ]; then
    echo "[ERROR] The scripts for TDNN currently do not support IHM data."
    exit 1;
  else
    # The following script cleans the data and produces cleaned data
    # in data/$mic/$MODE/train_cleaned, and a corresponding system
    # in exp/$mic/$MODE/tri3_cleaned.  It also decodes.
    #
    # Note: local/run_cleanup_segmentation.sh defaults to using 50 jobs,
    # you can reduce it using the --nj option if you want.
    local/run_cleanup_segmentation.sh --mic $mic  --mode $MODE
  fi
fi

if [ $stage -le 12 ]; then
  echo "Advanced system: TDNN - training."

  ali_opt=
  [ "$mic" != "ihm" ] && ali_opt="--use-ihm-ali true"
  local/chain/run_tdnn.sh $ali_opt --mic $mic  --mode $MODE

  echo "TDNN done."
fi




# LSTM. As an advanced system by default it is not constructed.
if [ $stage -le 12 ]; then
  echo "Advanced system: LSTM."

fi 





