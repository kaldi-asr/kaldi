#!/bin/bash

cmd="run.pl"
nj=20
stage=0

# https://github.com/pyannote/pyannote-audio-hub#overlapped-speech-detection
model=ovl_dihard  # Select ovl_dihard or ovl_ami

. parse_options.sh || exit 1;
echo "$0 $@"  # Print the command line for logging
if [ $# -ne 2 ]; then
  echo "This script performs overlap detection using the pretrained model from "
  echo "pyannote audio hub. The overlap detection is done on whole recordings,"
  echo "i.e., segment file, if present, is ignored."
  echo "Usage: $0 <data-dir> <output-dir>"
  echo " e.g.: $0 data/dev exp/dev_overlap"
  exit 1
fi

data_dir=$1
dir=$2

mkdir -p $dir

# check if PyTorch is installed (used for pyannote audio hub)
result=`$HOME/miniconda3/bin/python -c "\
try:
    import torch
    import pyannote.audio
    print('1')
except ImportError:
    print('0')"`

if [ "$result" == "1" ]; then
    echo "Pyannote and PyTorch are installed"
else
    echo "PyTorch/Pyannote is not installed. Please install using `source ${miniconda_dir}/bin/activate; pip install torch pyannote.audio`"
    exit 1
fi

if [ $stage -le 0 ]; then
  # split wav.scp for faster processing
  nj_wav=$(wc -l < ${data_dir}/wav.scp)
  nj=$((nj>nj_wav ? nj_wav : nj))
  echo "$0: Splitting wav.scp into ${nj} parts"
  split -n l/${nj} -a 2 --numeric-suffixes=1 ${data_dir}/wav.scp ${dir}/wav.scp.
  for f in ${dir}/wav.scp.[0-9]*; do
      mv "$f" "${f/scp\.0/scp\.}"
  done
fi

if [ $stage -le 1 ]; then
  echo "$0: Detecting overlaps..."
  $cmd JOB=1:$nj ${dir}/log/overlap.JOB.log \
    $HOME/miniconda3/bin/python steps/overlap/detect_overlaps_pyannote.py  \
      --model-name ${model} \
      ${dir}/wav.scp.JOB ${dir}
fi

for f in ${dir}/*.rttm; do
  file_id=$(basename $f .rttm)
  cat $f | awk -v file=$file_id '{$2=file;$8="overlap"}{print $0}'
done >$dir/rttm
exit 0
