#!/bin/bash

# Copyright 2016  Pegah Ghahremani
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
compress=true
compress_method=3
raw_config=conf/raw.conf
write_utt2num_frames=false  # if true writes utt2num_frames
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir> <path-to-segmented-dir>";
   echo "e.g.: $0 data/train exp/make_segment/train segments"
   echo "options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  featdir=$3
else
  featdir=$data/data
fi

# make $featdir an absolute pathname.
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`
scp=$data/wav.scp

mkdir -p $featdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

required="$scp $raw_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $featdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $featdir/raw_wav_feat_$name.$n.ark
done

if $write_utt2num_frames; then
  write_num_frames_opt="--write-num-frames=ark,t:$logdir/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."

  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_raw_feats_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
      compute-raw-frame-feats --config=$raw_config ark:- ark:- \| \
      copy-feats $write_num_frames_opt \
      --compress=$compress --compression-method=$compress_method ark:- \
      ark,scp:$featdir/raw_wav_feat_$name.JOB.ark,$featdir/raw_wav_feat_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;


  # add ,p to the input rspecifier so that we can just skip over
  # utterances that have bad wave data.

  $cmd JOB=1:$nj $logdir/make_mfcc_${name}.JOB.log \
    compute-raw-frame-feats --config=$raw_config \
     scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \
      copy-feats $write_num_frames_opt --compress=$compress \
      --compression-method=$compress_method ark:- \
      ark,scp:$featdir/raw_wav_feat_$name.JOB.ark,$featdir/raw_wav_feat_$name.JOB.scp \
      || exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featdir/raw_wav_feat_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/raw_wav_feat_${name}.*.scp  $logdir/segments.* 2>/dev/null
echo "Successed generate $data"
