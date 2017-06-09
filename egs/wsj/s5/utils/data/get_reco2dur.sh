#!/bin/bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script operates on a data directory, such as in data/train/, and adds the
# reco2dur file if it does not already exist.  The file 'reco2dur' maps from
# utterance to the duration of the utterance in seconds.  This script works it
# out from the 'segments' file, or, if not present, from the wav.scp file (it
# first tries interrogating the headers, and if this fails, it reads the wave
# files in entirely.)

frame_shift=0.01
cmd=run.pl
nj=4

. utils/parse_options.sh
. ./path.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <datadir>"
  echo "e.g.:"
  echo " $0 data/train"
  echo " Options:"
  echo " --frame-shift      # frame shift in seconds. Only relevant when we are"
  echo "                    # getting duration from feats.scp (default: 0.01). "
  exit 1
fi

export LC_ALL=C

data=$1

if [ -s $data/reco2dur ] && \
  [ $(cat $data/wav.scp | wc -l) -eq $(cat $data/reco2dur | wc -l) ]; then
  echo "$0: $data/reco2dur already exists with the expected length.  We won't recompute it."
  exit 0;
fi

# if the wav.scp contains only lines of the form
# utt1  /foo/bar/sph2pipe -f wav /baz/foo.sph |
if cat $data/wav.scp | perl -e '
   while (<>) { s/\|\s*$/ |/;  # make sure final | is preceded by space.
           @A = split; if (!($#A == 5 && $A[1] =~ m/sph2pipe$/ &&
                             $A[2] eq "-f" && $A[3] eq "wav" && $A[5] eq "|")) { exit(1); }
           $utt = $A[0]; $sphere_file = $A[4];

           if (!open(F, "<$sphere_file")) { die "Error opening sphere file $sphere_file"; }
           $sample_rate = -1;  $sample_count = -1;
           for ($n = 0; $n <= 30; $n++) {
              $line = <F>;
              if ($line =~ m/sample_rate -i (\d+)/) { $sample_rate = $1; }
              if ($line =~ m/sample_count -i (\d+)/) { $sample_count = $1; }
              if ($line =~ m/end_head/) { break; }
           }
           close(F);
           if ($sample_rate == -1 || $sample_count == -1) {
             die "could not parse sphere header from $sphere_file";
           }
           $duration = $sample_count * 1.0 / $sample_rate;
           print "$utt $duration\n";
   } ' > $data/reco2dur; then
  echo "$0: successfully obtained utterance lengths from sphere-file headers"
else
  echo "$0: could not get utterance lengths from sphere-file headers, using wav-to-duration"
  if ! command -v wav-to-duration >/dev/null; then
    echo  "$0: wav-to-duration is not on your path"
    exit 1;
  fi

  read_entire_file=false
  if cat $data/wav.scp | grep -q 'sox.*speed'; then
    read_entire_file=true
    echo "$0: reading from the entire wav file to fix the problem caused by sox commands with speed perturbation. It is going to be slow."
    echo "... It is much faster if you call get_reco2dur.sh *before* doing the speed perturbation via e.g. perturb_data_dir_speed.sh or "
    echo "... perturb_data_dir_speed_3way.sh."
  fi

  utils/split_data.sh $data $nj
  if ! $cmd JOB=1:$nj $data/log/get_wav_duration.JOB.log wav-to-duration --read-entire-file=$read_entire_file scp:$data/split$nj/JOB/wav.scp ark,t:$data/split$nj/JOB/reco2dur 2>&1; then
    echo "$0: there was a problem getting the durations; moving $data/reco2dur to $data/.backup/"
    mkdir -p $data/.backup/
    mv $data/reco2dur $data/.backup/
    exit 1
  fi
  
  for n in `seq $nj`; do
    cat $data/split$nj/$n/reco2dur
  done > $data/reco2dur
fi

echo "$0: computed $data/reco2dur"

exit 0

