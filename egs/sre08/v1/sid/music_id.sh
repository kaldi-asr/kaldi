#!/usr/bin/env bash

# Copyright    2015  David Snyder
# Apache 2.0.

# This script calculates the relative probability of music versus
# speech.

# Begin configuration section.
nj=10
cmd="run.pl"
stage=-4
num_gselect=20 # Gaussian-selection using diagonal and full covariance models
norm_vars=false
center=true
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 <music-ubm-dir> <speech-ubm-dir> <data> <exp-dir>"
  echo " e.g.: $0  exp/full_ubm_music exp/full_ubm_speech data/test exp/test_results"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --cleanup <true,false|true>                      # If true, clean up temporary files"
  echo "  --num-processes <n|4>                            # Number of processes for each queue job (relates"
  echo "                                                   # to summing accs in memory)"
  echo "  --stage <stage|-4>                               # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  exit 1;
fi

music_ubmdir=$1
speech_ubmdir=$2
data=$3
dir=$4

delta_opts=`cat $speech_ubmdir/delta_opts 2>/dev/null`

for f in $music_ubmdir/final.ubm $speech_ubmdir/final.ubm $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log || exit 1;
sdata=$data/split$nj
utils/split_data.sh $data $nj || exit 1;

## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=$norm_vars --center=$center --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

if [ $stage -le -2 ]; then
  $cmd $dir/log/music_convert.log \
    fgmm-global-to-gmm $music_ubmdir/final.ubm $dir/music_final.dubm || exit 1;
fi
if [ $stage -le -2 ]; then
  $cmd $dir/log/speech_convert.log \
    fgmm-global-to-gmm $speech_ubmdir/final.ubm $dir/speech_final.dubm || exit 1;
fi

# Do Gaussian selection using the diagonal forms of the models.

if [ $stage -le -1 ]; then
  echo $nj > $dir/num_jobs
  echo "$0: doing Gaussian selection for music UBM"
  $cmd JOB=1:$nj $dir/log/music_gselect.JOB.log \
    gmm-gselect --n=$num_gselect $dir/music_final.dubm "$feats" ark:- \| \
    fgmm-gselect --gselect=ark,s,cs:- --n=$num_gselect $music_ubmdir/final.ubm \
      "$feats" "ark:|gzip -c >$dir/music_gselect.JOB.gz" || exit 1;

  echo $nj > $dir/num_jobs
  echo "$0: doing Gaussian selection for speech UBM"
  $cmd JOB=1:$nj $dir/log/speech_gselect.JOB.log \
    gmm-gselect --n=$num_gselect $dir/speech_final.dubm "$feats" ark:- \| \
    fgmm-gselect --gselect=ark,s,cs:- --n=$num_gselect $speech_ubmdir/final.ubm \
      "$feats" "ark:|gzip -c >$dir/speech_gselect.JOB.gz" || exit 1;
fi

if ! [ $nj -eq $(cat $dir/num_jobs) ]; then
  echo "Number of jobs mismatch"
  exit 1;
fi

# Calculate the average frame-level log-likelihoods for the utterances under
# the music and speech UBMs.
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/get_music_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=true \
     "--gselect=ark,s,cs:gunzip -c $dir/music_gselect.JOB.gz|" $music_ubmdir/final.ubm \
      "$feats" ark,t:$dir/music_logprob.JOB || exit 1;
fi
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/get_speech_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=true \
     "--gselect=ark,s,cs:gunzip -c $dir/speech_gselect.JOB.gz|" $speech_ubmdir/final.ubm \
      "$feats" ark,t:$dir/speech_logprob.JOB || exit 1;
fi

if [ $stage -le 2 ]; then

  for j in $(seq $nj); do cat $dir/music_logprob.$j; done > $dir/music_logprob
  for j in $(seq $nj); do cat $dir/speech_logprob.$j; done > $dir/speech_logprob

  n1=$(cat $dir/music_logprob | wc -l)
  n2=$(cat $dir/speech_logprob | wc -l)

  if [ $n1 -ne $n2 ]; then
    echo "Number of lines mismatch, music versus speech UBM probs: $n1 vs $n2"
    exit 1;
  fi

  paste $dir/music_logprob $dir/speech_logprob | \
    awk '{if ($1 != $3) { print >/dev/stderr "Sorting mismatch"; exit(1);  } print $1, $2, $4;}' \
    >$dir/logprob || exit 1;

  cat $dir/logprob | \
    awk '{lratio = $2-$3; print $1, 1/(1+exp(-lratio));}' \
    >$dir/ratio || exit 1;
fi

if $cleanup; then
  rm $dir/speech_gselect.*.gz
  rm $dir/music_gselect.*.gz
fi

exit 0;
