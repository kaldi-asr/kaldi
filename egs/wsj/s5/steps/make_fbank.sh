#!/bin/bash

# Copyright 2012-2016  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
fbank_config=conf/fbank.conf
compress=true
write_utt2num_frames=false  # if true writes utt2num_frames
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]";
   echo "e.g.: $0 data/train exp/make_fbank/train mfcc"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --fbank-config <config-file>                     # config passed to compute-fbank-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  fbankdir=$3
else
  fbankdir=$data/data
fi


# make $fbankdir an absolute pathname.
fbankdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $fbankdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $fbankdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $fbank_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_fbank.sh: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi

for n in $(seq $nj); do
  # the next command does nothing unless $fbankdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $fbankdir/raw_fbank_$name.$n.ark
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

  $cmd JOB=1:$nj $logdir/make_fbank_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    compute-fbank-feats $vtln_opts --verbose=2 --config=$fbank_config ark:- ark:- \| \
    copy-feats --compress=$compress $write_num_frames_opt ark:- \
     ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/make_fbank_${name}.JOB.log \
    compute-fbank-feats $vtln_opts --verbose=2 --config=$fbank_config scp,p:$logdir/wav.JOB.scp ark:- \| \
    copy-feats --compress=$compress $write_num_frames_opt ark:- \
     ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
     || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing fbank features for $name:"
  tail $logdir/make_fbank_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $fbankdir/raw_fbank_$name.$n.scp || exit 1;
done > $data/feats.scp

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $logdir/utt2num_frames.$n || exit 1;
  done > $data/utt2num_frames || exit 1
  rm $logdir/uttnum_frames.*
fi

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
