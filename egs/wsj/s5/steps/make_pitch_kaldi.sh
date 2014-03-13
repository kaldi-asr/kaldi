#!/bin/bash 

# Copyright 2013  Johns Hopkins University (Author: Pegah Ghahremani, Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
debug_file=
pitch_config=conf/pitch.conf
postprocess_config=
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_pitch_kaldi.sh [options] <data-dir> <log-dir> <path-to-pitchdir>";
   echo "options: "
   echo "  --pitch-config <config-file>                     # config passed to compute-kaldi-pitch-feats "
   echo "  --postprocess-config <config-file>               # config passed to process-kaldi-pitch-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --debug-file <file>                              # name of matlab file to write some diagnostic"
   echo "                                                   # information to (about distribution of outputs)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
pitchdir=$3


# make $pitchdir an absolute pathname.
pitchdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $pitchdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $pitchdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp

required="$scp $pitch_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_pitch_new.sh: no such file $f"
    exit 1;
  fi
done

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.


if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for ((n=1; n<=nj; n++)); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  if [ ! -z "$postprocess_config" ]; then
    postprocess_config_opt="--config=$postprocess_config";
  else
    postprocess_config_opt=
  fi

  $cmd JOB=1:$nj $logdir/make_pitch.JOB.log \
    extract-segments scp:$scp $logdir/segments.JOB ark:- \| \
    compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config ark:- ark:- \| \
    process-kaldi-pitch-feats $postprocess_config_opt ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$pitchdir/pitch_$name.JOB.ark,$pitchdir/pitch_$name.JOB.scp \
      || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for ((n=1; n<=nj; n++)); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
 
  $cmd JOB=1:$nj $logdir/make_pitch.JOB.log \
    compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config scp:$logdir/wav_${name}.JOB.scp ark:- \| \
    process-kaldi-pitch-feats $postprocess_config_opt ark:- \
      ark,scp:$pitchdir/pitch_$name.JOB.ark,$pitchdir/pitch_$name.JOB.scp \
      || exit 1;
fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing pitch features for $name:"
  tail $logdir/make_pitch.*.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $pitchdir/pitch_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating Kaldi pitch features for $name"

if [ ! -z "$debug_file" ]; then
  echo "A = [" > $debug_file
  copy-feats scp:$data/feats.scp ark,t:- | grep -v ']' | grep -v '\[' | awk '{if (NF == 2) { print; }}' | head -n 200000 \
    >> $debug

  cat <<'EOF' >>$debug
];
pov = A(:, 1);
pitch = A(:, 2);
subplot(2, 2, 1);
hist(pov, 30);
legend('pov')
subplot(2, 2, 2);
hist(pitch, 30);
legend('pitch')

len=size(pov, 1);
povD = pov(1:len-1) - pov(2:len);
subplot(2, 2, 3);
hist(povD, 30);
legend('delta-pov')

pitchD = pitch(1:len-1) - pitch(2:len);
pitchD = max(pitchD, -0.05);
pitchD = min(pitchD, 0.05);
subplot(2, 2, 4);
hist(pitchD, 50);
legend('delta-pitch');

print -deps 'C.eps'
EOF

fi
