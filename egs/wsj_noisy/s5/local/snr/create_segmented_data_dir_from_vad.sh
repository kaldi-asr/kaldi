#!/bin/bash

set -o pipefail
set -e 
set -u

. path.sh

cmd=run.pl
nj=4
feats=

. utils/parse_options.sh 

if [ $# -ne 6 ]; then
  echo "Usage: $0 <data-dir> <segments-file> <seg2utt-file> <feats-scp> <tmp-dir> <feat-dir> <out-data-dir>"
  echo " e.g.: $0 data/train_100k exp/vad_data_prep_train_100k/file_vad/segments exp/vad_data_prep_train_100k/file_vad/seg2utt exp/frame_snrs_a_train_100k/snr.scp exp/make_segmented_feats/log segmented_feats data/train_100k_seg"
  exit 1
fi

data=$1
segments=$2
seg2utt_file=$3
tmpdir=$4
featdir=$5
dir=$6


utils/copy_data_dir.sh --extra-files utt2uniq $data $dir

if [ -z "$feats" ]; then
  feats=$data/feats.scp
  cp $data/cmvn.scp $dir
fi

for f in $data/spk2utt $segments $seg2utt_file $feats; do
  if [ ! -f $f ]; then
    echo "$0: Could not read file $f"
    exit 1
  fi
done

featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

mkdir -p $tmpdir $featdir

rm -f $dir/{cmvn.scp,feats.scp,utt2spk,spk2utt,utt2uniq,spk2utt,text}
utils/filter_scp.pl -f 2 $data/spk2utt $segments > $dir/segments

awk '{print $1" "$2}' $dir/segments > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $featdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$featdir $featdir/storage
fi

data_id=`basename $data`

for n in `seq $nj`; do
  utils/create_data_link.pl $featdir/storage/raw_feats_${data_id}.$n.ark
done

$cmd JOB=1:$nj $tmpdir/extract_feature_segments.JOB.log \
  extract-feature-segments scp:$feats \
  "ark,t:utils/split_scp.pl -j $nj \$[JOB-1] $dir/segments |" \
  ark:- \| copy-feats --compress=true ark:- \
  ark,scp:$featdir/raw_feats_${data_id}.JOB.ark,$featdir/raw_feats_${data_id}.JOB.scp

for n in `seq $nj`; do 
  cat $featdir/raw_feats_${data_id}.$n.scp 
done | sort -k1,1 > $dir/feats.scp 

utils/fix_data_dir.sh $dir

