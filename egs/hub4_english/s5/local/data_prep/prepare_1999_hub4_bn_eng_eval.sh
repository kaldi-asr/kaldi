#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares 1999 HUB4 Broadcast News Evaluation English Test Material
# https://catalog.ldc.upenn.edu/LDC2000S88

set -e 
set -o pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo "$0 /export/corpora5/LDC/LDC2000S88/hub4_1999 data/local/data/eval99"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

mkdir -p $dir

if [ ! -d $SOURCE_DIR/bnews_99/ ]; then
  echo "$0: Invalid SOURCE-DIR for LDC2000S88 corpus"
  exit 1
fi

export PATH=$PATH:$KALDI_ROOT/tools/sph2pipe_v2.5
sph2pipe=`which sph2pipe` || { echo "sph2pipe not found in PATH."; exit 1; }

for f in bn99en_1 bn99en_2; do 
  if [ "$f" == "bn99en_1" ]; then
    affix=eval99_1
  else
    affix=eval99_2
  fi

  python -c '
import sys, os
sys.path.insert(0, "local/data_prep")
import hub4_utils
uem = sys.argv[1]
reco, ext = os.path.splitext(os.path.basename(uem))
for line in open(uem).readlines():
  line = hub4_utils.parse_uem_line(reco, line)
  if line is not None:
    print (line)' $SOURCE_DIR/bnews_99/$f.uem > $dir/${affix}_uem_segments

  awk '{print  $1" "$2}' $dir/${affix}_uem_segments > $dir/${affix}_uem_utt2spk

  cat $SOURCE_DIR/bnews_99/$f.seg | \
    python -c '
import sys
sys.path.insert(0, "local/data_prep")
import hub4_utils
with open(sys.argv[1], "w") as s_f, open(sys.argv[2], "w") as u_f:
  for line in sys.stdin.readlines():
    tup = hub4_utils.parse_cmu_seg_line(line)
    if tup is not None:
      segments_line, utt2spk_line = tup
      s_f.write("{0}\n".format(segments_line))
      u_f.write("{0}\n".format(utt2spk_line))' \
        $dir/${affix}_pem_segments $dir/${affix}_pem_utt2spk
  
  echo "$f $sph2pipe -f wav $SOURCE_DIR/bnews_99/$f.sph |" > ${dir}/${affix}_wav_scp
done 

cp $SOURCE_DIR/bnews_99/en981118.glm $dir/eval99_1_glm
cp $SOURCE_DIR/bnews_99/bn99en_1.stm $dir/eval99_1_stm

cp $SOURCE_DIR/bnews_99/en991231.glm $dir/eval99_2_glm
cp $SOURCE_DIR/bnews_99/bn99en_2.stm $dir/eval99_2_stm
