#!/bin/bash

# Copyright 2016  Matthew Maciejewski
# Apache 2.0.

if [ $# -ne 3 ]; then
  echo "Usage: make_callhome_test.sh <sph-dir> <seg-dir> <data-dir>"
  echo "  eg: make_callhome_test.sh /home/dpovey/diarization/data /home/dpovey/diarization/chome.v0 data/callhome"
  exit 1;
fi

dir=$PWD/$3

. ./cmd.sh
. ./path.sh

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

mkdir -p $dir

# Create wav.scp
ls $1 | cut -f1 -d'.' | awk '{printf("%s '$sph2pipe' -f wav '$1'/%s.sph |\n", $1, $1)}' > $dir/wav.scp

# Create segments, utt2spk, seg2spk, utt2num
rm -f $dir/{segments,utt2spk,seg2spk,utt2num}
for uttid in $(ls $2 | cut -f1 -d'.'); do
  awk 'BEGIN{i=0}{printf("'$uttid'_%03d '$uttid' %s %s\n", i++, $1, $2)}' < $2/$uttid.ref >> $dir/segments
  awk 'BEGIN{i=0}{printf("'$uttid'_%03d '$uttid'\n", i++)}' < $2/$uttid.ref >> $dir/utt2spk
  awk 'BEGIN{i=0}{printf("'$uttid'_%03d %s\n", i++, $3)}' < $2/$uttid.ref >> $dir/seg2spk
  echo "$uttid $(cat $2/$uttid.ref | cut -f3 -d' ' | sort -u | wc -l)" >> $dir/utt2num
done

# Create spk2utt from utt2spk
cat $dir/utt2spk | utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Verify completion
utils/validate_data_dir.sh --no-text --no-feats $dir
utils/fix_data_dir.sh $dir
