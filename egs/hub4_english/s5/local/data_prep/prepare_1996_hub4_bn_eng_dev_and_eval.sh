#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares 1996 English Broadcast News Dev and Eval (HUB4)
# https://catalog.ldc.upenn.edu/LDC97S66

set -e
set -o pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo "$0 /export/corpora/LDC/LDC97S66/1996_eng_bcast_dev_eval data/local/data/hub4_96_dev_eval"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

mkdir -p $dir

for d in $SOURCE_DIR/dev/devdata $SOURCE_DIR/eval/evaldata; do 
  if [ ! -d $d ]; then
    echo "$0: Invalid SOURCE-DIR $SOURCE_DIR for LDC97S66 corpus"
    exit 1
  fi
done

for d in dev eval; do 
  if [ $d == "dev" ]; then
    suffix=dt
  else
    suffix=ev
  fi

  python -c '
import sys, os
sys.path.insert(0, "local/data_prep")
import hub4_utils
uem = sys.argv[1]
for line in open(uem).readlines():
  line = hub4_utils.parse_uem_line(None, line)
  if line is not None:
    print (line)' $SOURCE_DIR/${d}/${d}data/h496${suffix}.uem > $dir/${d}96_uem_segments
  awk '{print $1" "$2}' $dir/${d}96_uem_segments > $dir/${d}96_uem_utt2spk
done 

for d in dev eval; do 
  if [ $d == "dev" ]; then
    suffix=dt
  else
    suffix=ev
  fi

  cat $SOURCE_DIR/${d}/${d}data/h496${suffix}.pem | \
    python -c '
import sys
sys.path.insert(0, "local/data_prep")
import hub4_utils
with open(sys.argv[1], "w") as s_f, open(sys.argv[2], "w") as u_f:
  for line in sys.stdin.readlines():
    tup = hub4_utils.parse_cmu_seg_line(line, prepend_reco_to_spk=True)
    if tup is not None:
      segments_line, utt2spk_line = tup
      s_f.write("{0}\n".format(segments_line))
      u_f.write("{0}\n".format(utt2spk_line))' \
        $dir/${d}96_pem_segments $dir/${d}96_pem_utt2spk
done
 
export PATH=$PATH:$KALDI_ROOT/tools/sph2pipe_v2.5
sph2pipe=`which sph2pipe` || { echo "sph2pipe not found in PATH."; exit 1; }

for x in `ls $SOURCE_DIR/dev/devdata/*.sph`; do
  y=`basename $x`
  z=${y%.sph}
  echo "$z $sph2pipe -f wav $x |";
done > $dir/dev96_wav_scp

cat $dir/dev96_pem_segments | awk '{print $2}' | \
  utils/filter_scp.pl /dev/stdin $dir/dev96_wav_scp > $dir/dev96_pem_wav_scp
cat $dir/dev96_uem_segments | awk '{print $2}' | \
  utils/filter_scp.pl /dev/stdin $dir/dev96_wav_scp > $dir/dev96_uem_wav_scp

for x in `ls $SOURCE_DIR/eval/evaldata/*.sph`; do
  y=`basename $x`
  z=${y%.sph}
  echo "$z $sph2pipe -f wav $x |";
done > $dir/eval96_wav_scp

cp $SOURCE_DIR/eval/evaldata/et96_1.glm $dir/glm

cp $SOURCE_DIR/eval/evaldata/et96_1.utm $dir/eval96_utm
cp $SOURCE_DIR/dev/devdata/et96_1.utm $dir/dev96_utm

cp $SOURCE_DIR/eval/evaldata/h496ev.stm $dir/eval96_stm

cp $SOURCE_DIR/dev/devdata/h496dtpe.stm $dir/dev96_pem_stm
cp $SOURCE_DIR/dev/devdata/h496dtue.stm $dir/dev96_uem_stm
