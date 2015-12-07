#!/bin/bash

set -o pipefail
set -e 
set -u

. path.sh

cmd=run.pl
nj=4

. utils/parse_options.sh 

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-dir> <vad-dir> <tmp-dir> <dir>"
  echo " e.g.: $0 data/train_100k exp/vad_data_prep_train_100k/file_vad exp/make_segmented_vad/train_100k exp/vad_data_prep_train_100k/seg_vad"
  exit 1
fi

data=$1
vad_dir=$2
tmpdir=$3
dir=$4

segments=$vad_dir/segments
seg2utt_file=$vad_dir/reco2utt
vad_scp=$vad_dir/vad.scp

for f in $vad_scp $segments $seg2utt_file; do
  if [ ! -f $f ]; then
    echo "$0: Could not read file $f"
    exit 1
  fi
done

mkdir -p $dir/split$nj

utils/filter_scp.pl -f 2 $data/utt2spk $segments > $dir/segments
utils/filter_scp.pl -f 2 $data/utt2spk $seg2utt_file > $dir/reco2utt

$cmd JOB=1:$nj $tmpdir/extract_vad_segments.JOB.log \
  extract-int-vector-segments scp:$vad_scp \
  "ark,t:utils/split_scp.pl -j $nj \$[JOB-1] $dir/segments |" \
  ark:- \| segmentation-init-from-ali ark:- ark:- \| \
  segmentation-post-process --merge-labels=0:2 --merge-dst-label=0 ark:- ark:- \| \
  segmentation-to-ali ark:- ark,scp:$dir/split$nj/vad.JOB.ark,$dir/split$nj/vad.JOB.scp || exit 1

for n in `seq $nj`; do 
  cat $dir/split$nj/vad.$n.scp 
done | sort -k1,1 > $dir/vad.scp

