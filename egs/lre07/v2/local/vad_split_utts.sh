#!/bin/bash

max_voiced=3000
stage=0
cleanup=true

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <in-data-dir> <split-mfcc-out-dir> <out-data-dir>"
  echo "e.g.: $0 --max-voiced 3000 data/train mfcc data/train_split"
  echo "This script splits up long utterances into smaller pieces."
  echo "It assumes the wav.scp contains has a certain form, with .sph"
  echo "files in it (so the script is not completely general)."
  exit 1;
fi

in_dir=$1
mfccdir=$2
dir=$3

for f in $in_dir/{utt2spk,spk2utt,wav.scp,utt2lang,feats.scp,vad.scp}; do
  if [ ! -f $f ]; then
    echo "$0: expected input file $f to exist";
    exit 1;
  fi
done

if [ $stage -le 0 ]; then
  utils/validate_data_dir.sh --no-text $in_dir || exit 1;
  mkdir -p $dir/temp || exit 1;
fi

if [ $stage -le 1 ]; then

create-split-from-vad --max-voiced=$max_voiced scp:$in_dir/vad.scp $dir/frame_indexed_segments;

extract-rows $dir/frame_indexed_segments scp:$in_dir/feats.scp ark,scp:$mfccdir/raw_mfcc_split.ark,$dir/feats.scp;

copy-vector-segments $dir/frame_indexed_segments scp:$in_dir/vad.scp ark,scp:$mfccdir/vad_split.ark,$dir/temp/vad.scp;
sort $dir/temp/vad.scp > $dir/vad.scp;
fi

if [ $stage -le 2 ]; then
local/vad_split_utts_fix_data.pl $in_dir $dir;
fi

utils/filter_scp.pl -f 0 \
<(echo "`awk < "$dir/segments" '{ print $2 }'`") $in_dir/wav.scp \
 > $dir/wav.scp

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
utils/validate_data_dir.sh --no-text --no-feats $dir || exit 1;

$cleanup && rm -r $dir/temp

exit 0;
