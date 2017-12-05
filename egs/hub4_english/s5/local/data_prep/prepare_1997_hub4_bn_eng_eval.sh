#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares 1997 HUB4 English Evaluation corpus
# https://catalog.ldc.upenn.edu/LDC2002S11

set -e
set -o pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo "$0 /export/corpora/LDC/LDC2002S11/hub4e_97 data/local/data/eval97"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

mkdir -p $dir

if [ ! -d $SOURCE_DIR/h4e_evl/ ]; then
  echo "$0: Invalid SOURCE-DIR $SOURCE_DIR for LDC2002S11 corpus"
  exit 1
fi

for uem in $SOURCE_DIR/h4e_evl/h4e_97.uem; do
  python -c '
import sys, os
sys.path.insert(0, "local/data_prep")
import hub4_utils
uem = sys.argv[1]
reco, ext = os.path.splitext(os.path.basename(uem))
for line in open(uem).readlines():
  line = hub4_utils.parse_uem_line(reco, line)
  if line is not None:
    print (line)' $uem
done > $dir/segments
awk '{print $1" "$2}' $dir/segments > $dir/utt2spk

cat $SOURCE_DIR/h4e_evl/h4e_97.seg | \
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
        $dir/segments.pem $dir/utt2spk.pem
 
export PATH=$PATH:$KALDI_ROOT/tools/sph2pipe_v2.5
sph2pipe=`which sph2pipe` || { echo "sph2pipe not found in PATH."; exit 1; }
for x in `ls $SOURCE_DIR/h4e_evl/*.sph`; do
  y=`basename $x`
  z=${y%.sph}
  echo "$z $sph2pipe -f wav $x |";
done > $dir/wav.scp

cp $SOURCE_DIR/h4e_evl/h4e_97_1.glm $dir/glm
cp $SOURCE_DIR/h4e_evl/h4e_97.stm $dir/stm
