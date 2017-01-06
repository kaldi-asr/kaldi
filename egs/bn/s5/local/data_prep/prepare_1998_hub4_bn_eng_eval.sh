#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo "local/prepare_1998_hub4_bn_eng_eval.sh /export/corpora/LDC/LDC2000S86/ data/local/data/eval98"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

mkdir -p $dir

for uem in $SOURCE_DIR/h4e_evl/h4e_98_{1,2}.uem; do
  python -c '
import sys, os
uem = sys.argv[1]
reco, ext = os.path.splitext(os.path.basename(uem))
for line in open(uem).readlines():
  line = line.strip()
  if len(line) == 0 or line[0:2] == ";;":
    continue
  parts = line.split()

  assert parts[1] == "1"
  start_time = float(parts[2])
  end_time = float(parts[3])
  
  utt = "{0}-{1:06d}-{2:06d}".format(reco, int(start_time * 100), 
                                     int(end_time * 100))
  print ("{0} {1} {2} {3}".format(utt, reco, start_time, end_time))' $uem 
done > $dir/segments

cat $SOURCE_DIR/h4e_evl/h4e_98_{1,2}.seg | \
  python -c '
from __future__ import print_function
import sys

segments_handle = open(sys.argv[1], "w")
utt2spk_handle = open(sys.argv[2], "w")
for line in sys.stdin.readlines():
  line = line.strip()
  if len(line) == 0 or line[0:2] == ";;":
    continue
  parts = line.split()

  reco = parts[0] 
  assert parts[1] == "1"
  spk = parts[2]
  start_time = float(parts[3])
  end_time = float(parts[4])
  
  utt = "{spk}-{0}-{1:06d}-{2:06d}".format(reco, int(start_time * 100), 
                                           int(end_time * 100), spk=spk)

  print ("{0} {1} {2} {3}".format(utt, reco, start_time, end_time),
         file=segments_handle)
  print ("{0} {1}".format(utt, spk),
         file=utt2spk_handle)
segments_handle.close()
utt2spk_handle.close()
' $dir/segments.pem $dir/utt2spk.pem
 
export PATH=$PATH:$KALDI_ROOT/tools/sph2pipe_v2.5
sph2pipe=`which sph2pipe` || { echo "sph2pipe not found in PATH."; exit 1; }
for x in `ls $SOURCE_DIR/h4e_evl/*.sph`; do
  y=`basename $x`
  z=${y%.sph}
  echo "$z $sph2pipe -f wav $x |";
done > $dir/wav.scp

awk '{print $1" "$1" 1"}' $dir/wav.scp > $dir/reco2file_and_channel

cp $SOURCE_DIR/h4e_evl/h4e_98.glm $dir/glm
cp $SOURCE_DIR/h4e_evl/h4e_98.stm $dir/stm

awk '{print $1" "$2}' $dir/segments > $dir/utt2spk

utils/fix_data_dir.sh $dir
utils/copy_data_dir.sh $dir ${dir}.pem

cp $dir/segments.pem ${dir}.pem/segments
cp $dir/utt2spk.pem ${dir}.pem/utt2spk
utils/fix_data_dir.sh ${dir}.pem
