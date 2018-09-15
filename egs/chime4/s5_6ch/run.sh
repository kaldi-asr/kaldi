# Kaldi ASR baseline for the CHiME-4 Challenge (6ch track: 6 channel track)
#
# Copyright 2016 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#           2017 JHU CLSP (Szu-Jui Chen)
#           2017 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

#####Baseline settings#####
# Usage: 
# Execute './run.sh' to get the models.
# We provide three kinds of beamform methods. Add option --enhancement blstm_gev, or --enhancement beamformit_5mics
# or --enhancement single_blstmmask to use them. i.g. './run.sh --enhancement blstm_gev'
#
# We stopped to support the old CHiME-3/4 baseline. If you want to reproduce the old results
# Please use the old version of Kaldi, e.g., git checkout 9e8ff73648917836d0870c8f6fdd2ff4bdde384f

# Config:
stage=0 # resume training with --stage N
enhancement=blstm_gev #### or your method
# if the following options are true, they wouldn't train a model again and will only do decoding
gmm_decode_only=false
tdnn_decode_only=false
# make it true when you want to add enhanced data into training set. But please note that when changing enhancement method,
# you may need to retrain from run_gmm.sh and avoid using decode-only options above
add_enhanced_data=true

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#####check data and model paths################
# Set a main root directory of the CHiME4 data
# If you use scripts distributed in the CHiME4 package,
chime4_data=`pwd`/../..
# Otherwise, please specify it, e.g.,
# chime4_data=/db/laputa1/data/processed/public/CHiME4
# chime3_data=/data2/archive/speech-db/original/public/CHiME3

case $(hostname -f) in
  *.clsp.jhu.edu) 
      chime4_data=/export/corpora4/CHiME4/CHiME3 # JHU,
      chime3_data=/export/corpora5/CHiME3 
      ;;
esac 

if [ ! -d $chime4_data ]; then
  echo "$chime4_data does not exist. Please specify chime4 data root correctly" && exit 1;
fi
if [ ! -d $chime3_data ]; then
  echo "$chime3_data does not exist. Please specify chime4 data root correctly" && exit 1;
fi

#####main program start################
# You can execute run_init.sh only "once"
# This creates 3-gram LM, FSTs, and basic task files
if [ $stage -le 0 ]; then
  local/run_init.sh $chime4_data
fi

# Using Beamformit or mask-based beamformer
# note that beamformed WAV files are generated in the following directory
enhancement_data=`pwd`/enhan/$enhancement
if [ $stage -le 1 ]; then
   case $enhancement in
    beamformit_5mics)
        local/run_beamform_6ch_track.sh --cmd "$train_cmd" --nj 20 $chime4_data/data/audio/16kHz/isolated_6ch_track $enhancement_data
        ;;
    blstm_gev)
        local/run_blstm_gev.sh --cmd "$train_cmd" --nj 20 $chime4_data $chime3_data $enhancement_data 0
        ;;
    single_blstmmask)
        local/run_blstm_gev.sh --cmd "$train_cmd" --nj 20 $chime4_data $chime3_data $enhancement_data 5 
        ;;
    *)
        echo "Usage: --enhancement blstm_gev, or --enhancement beamformit_5mics , or --enhancement single_blstmmask" 
        exit 1;
   esac
fi

# Compute PESQ, STOI, eSTOI, and SDR scores
if [ $stage -le 2 ]; then
  if [ ! -f local/bss_eval_sources.m ] || [ ! -f local/stoi.m ] || [ ! -f local/estoi.m ] || [ ! -f local/PESQ ]; then
    # download and install speech enhancement evaluation tools
    local/download_se_eval_tool.sh
  fi
  chime4_rir_data=local/nn-gev/data/audio/16kHz/isolated_ext
  if [ ! -d $chime4_rir_data ]; then
    echo "$chime4_rir_data does not exist. Please run 'blstm_gev' enhancement method first;" && exit 1;
  fi
  local/compute_pesq.sh $enhancement $enhancement_data $chime4_rir_data $PWD
  local/compute_stoi_estoi_sdr.sh $enhancement $enhancement_data $chime4_rir_data
  local/compute_pesq.sh NOISY_1ch $chime4_data/data/audio/16kHz/isolated_1ch_track/ $chime4_rir_data $PWD
  local/compute_stoi_estoi_sdr.sh NOISY_1ch $chime4_data/data/audio/16kHz/isolated_1ch_track/ $chime4_rir_data
  local/write_se_results.sh $enhancement
  local/write_se_results.sh NOISY_1ch
fi

# GMM based ASR experiment
# Please set a directory of your speech enhancement method.
# The directory structure and audio files must follow the attached baseline enhancement directory
if [ $stage -le 3 ]; then
  local/run_gmm.sh --add-enhanced-data $add_enhanced_data \
    --decode-only $gmm_decode_only $enhancement $enhancement_data $chime4_data
fi

# TDNN based ASR experiment
# Since it takes time to evaluate TDNN, we make the GMM and TDNN scripts separately.
# You may execute it after you would have promising results using GMM-based ASR experiments
if [ $stage -le 4 ]; then
  local/chain/run_tdnn.sh --decode-only $tdnn_decode_only $enhancement
fi

# LM-rescoring experiment with 5-gram and RNN LMs
# It takes a few days to train a RNNLM.
if [ $stage -le 5 ]; then
  local/run_lmrescore_tdnn.sh $chime4_data $enhancement
fi

# LM-rescoring experiment with LSTM LMs
if [ $stage -le 6 ]; then
  local/rnnlm/run_lstm.sh $enhancement
fi

echo "Done."
