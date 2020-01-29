#!/usr/bin/env bash

# Copyright 2019, Johns Hopkins University (Author: Shinji Watanabe)
# Apache 2.0
#
# This script generates synchronized audio data across arrays by considering
# the frame dropping, clock drift etc. done by Prof. Jon Barker at University of
# Sheffield. This script first downloads the synchronization tool and generate
# the synchronized audios and corresponding JSON transcription files
# Note that
# 1) the JSON format is slightly changed from the original CHiME-5 one (simplified
# thanks to the synchronization)
# 2) it requires sox v.14.4.2 and Python 3.6.7
# Unfortunately, the generated files would be different depending on the sox
# and Python versions and to generate the exactly same audio files, this script uses
# the fixed versions of sox and Python installed in the miniconda instead of system ones

. ./cmd.sh
. ./path.sh

# Config:
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Wrong #arguments ($#, expected 2)"
  echo "Usage: local/generate_chime6_data.sh [options] <chime5-in-dir> <chime6-out-dir>"
  echo "main options (for others, see top of script file)"
  echo "  --cmd <cmd> # Command to run in parallel with"
  exit 1;
fi

sdir=$1
odir=$2
expdir=${PWD}/exp/chime6_data

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# get chime6-synchronisation tools
SYNC_PATH=${PWD}/chime6-synchronisation
if [ ! -d ${SYNC_PATH} ]; then
  git clone https://github.com/chimechallenge/chime6-synchronisation.git
fi

mkdir -p ${odir}
mkdir -p ${expdir}/log

# split the session to avoid too much disk access
sessions1="S01 S02 S03 S04 S05 S06 S07"
sessions2="S08 S09 S12 S13 S16 S17 S18"
sessions3="S19 S20 S21 S22 S23 S24"

CONDA_PATH=${HOME}/miniconda3/bin
IN_PATH=${sdir}/audio
OUT_PATH=${odir}/audio
TMP_PATH=${odir}/audio_tmp

if [ ! -d "${IN_PATH}" ]; then
  echo "please specify the CHiME-5 data path correctly"
  exit 1
fi
mkdir -p $OUT_PATH/train $OUT_PATH/eval $OUT_PATH/dev
mkdir -p $TMP_PATH/train $TMP_PATH/eval $TMP_PATH/dev

if [ -f ${odir}/audio/dev/S02_P05.wav ]; then
  echo "CHiME-6 date already exists"
  exit 0
fi

pushd ${SYNC_PATH}
echo "Correct for frame dropping"
for session in ${sessions1}; do
  $cmd ${expdir}/correct_signals_for_frame_drops.${session}.log \
    ${CONDA_PATH}/python correct_signals_for_frame_drops.py --session=${session} chime6_audio_edits.json $IN_PATH $TMP_PATH &
done
wait
for session in ${sessions2}; do
  $cmd ${expdir}/correct_signals_for_frame_drops.${session}.log \
    ${CONDA_PATH}/python correct_signals_for_frame_drops.py --session=${session} chime6_audio_edits.json $IN_PATH $TMP_PATH &
done
wait
for session in ${sessions3}; do
  $cmd ${expdir}/correct_signals_for_frame_drops.${session}.log \
    ${CONDA_PATH}/python correct_signals_for_frame_drops.py --session=${session} chime6_audio_edits.json $IN_PATH $TMP_PATH &
done
wait

echo "Sox processing for correcting clock drift"
for session in ${sessions1}; do
  $cmd ${expdir}/correct_signals_for_clock_drift.${session}.log \
    ${CONDA_PATH}/python correct_signals_for_clock_drift.py --session=${session} --sox_path $CONDA_PATH chime6_audio_edits.json $TMP_PATH $OUT_PATH &
done
wait
for session in ${sessions2}; do
  $cmd ${expdir}/correct_signals_for_clock_drift.${session}.log \
    ${CONDA_PATH}/python correct_signals_for_clock_drift.py --session=${session} --sox_path $CONDA_PATH chime6_audio_edits.json $TMP_PATH $OUT_PATH &
done
wait
for session in ${sessions3}; do
  $cmd ${expdir}/correct_signals_for_clock_drift.${session}.log \
    ${CONDA_PATH}/python correct_signals_for_clock_drift.py --session=${session} --sox_path $CONDA_PATH chime6_audio_edits.json $TMP_PATH $OUT_PATH &
done
wait

echo "adjust the JSON files"
mkdir -p ${odir}/transcriptions/eval ${odir}/transcriptions/dev ${odir}/transcriptions/train
${CONDA_PATH}/python correct_transcript_for_clock_drift.py --clock_drift_data chime6_audio_edits.json ${sdir}/transcriptions ${odir}/transcriptions
popd

# finally check md5sum
pushd ${odir}
echo "check MD5 hash value for generated audios"
md5sum -c ${SYNC_PATH}/audio_md5sums.txt || echo "check https://github.com/chimechallenge/chime6-synchronisation"
popd

echo "`basename $0` Done."
