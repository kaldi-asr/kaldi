#!/bin/bash

# Copyright 2015  University of Sheffield (Author: Ning Ma)
#           2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
# Apache 2.0.

# This script prepares the video features such that they match the audio data of CHiME-1/2.

# Begin configuration section.
nj=4
cmd=run.pl
audioRoot="./proc/wav"
videoRoot="./proc/video"

compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir> <feature-dir>";
   echo "e.g.:  $0 data/train exp/prepare_video/train video"
   echo "       $0 --audioRoot ./proc/wav --videoRoot ./proc/video data/train exp/prepare_video/train video"
   echo "options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --audioRoot <audio source directory>             # "
   echo "  --videoRoot <video source directory>             # "
   exit 1;
fi

data=$1
logdir=$2
featdir=$3

# make $featdir an absolute pathname.
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $featdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi
rm -rf $data/split*

scp=$data/wav.scp

required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_video.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

# create a modified copy of the wav.scp for the current data set by
# - replacing the audio root with the video root and,
# - adjusting the file extension from .wav to .ark
sed -e "s|${audioRoot}|${videoRoot}|g" $data/wav.scp | sed -e "s/.wav$/.ark/g" > $logdir/video.scp

# compute the byte offset of the data in the ark file based on the length of the utterance id
cat $logdir/video.scp | cut -f1 | awk '{ print (length($0)+1) }' > $logdir/video_offs

# create final scp file for video data
paste -d ":" $logdir/video.scp $logdir/video_offs > $logdir/video2.scp

# split video2.scp according to the number of jobs
split_scps=""
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/video.$n.scp"
done
utils/split_scp.pl $logdir/video2.scp $split_scps || exit 1;

# compress features and generate scp files
for n in $(seq $nj); do
  copy-feats --compress=$compress scp:$logdir/video.$n.scp \
    ark,scp:$featdir/raw_video_$name.$n.ark,$featdir/raw_video_$name.$n.scp \
    || exit 1;
done 2> $logdir/make_video_$name.log

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featdir/raw_video_$name.$n.scp || exit 1;
done > $data/feats.scp

#rm -f $featdir/ascii_video_$name.*.ark $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null
#rm -f $logdir/video.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded making video features for $name"
