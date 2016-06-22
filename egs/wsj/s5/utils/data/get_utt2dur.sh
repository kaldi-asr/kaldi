#!/bin/bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script operates on a data directory, such as in data/train/, and adds the
# utt2dur file if it does not already exist.  The file 'utt2dur' maps from
# utterance to the duration of the utterance in seconds.  This script works it
# out from the 'segments' file, or, if not present, from the wav.scp file (it
# first tries interrogating the headers, and if this fails, it reads the wave
# files in entirely.)

frame_shift=0.01

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

if [ -s $data/utt2dur ] && \
  [ $(cat $data/utt2spk | wc -l) -eq $(cat $data/utt2dur | wc -l) ]; then
  echo "$0: $data/utt2dur already exists with the expected length.  We won't recompute it."
  exit 0;
fi

if [ -f $data/segments ]; then
  echo "$0: working out $data/utt2dur from $data/segments"
  cat $data/segments | awk '{len=$4-$3; print $1, len;}' > $data/utt2dur
elif [ -f $data/wav.scp ]; then
  echo "$0: segments file does not exist so getting durations from wave files"

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
     } ' > $data/utt2dur; then
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
      echo "... It is much faster if you call get_utt2dur.sh *before* doing the speed perturbation via e.g. perturb_data_dir_speed.sh or "
      echo "... perturb_data_dir_speed_3way.sh."
    fi

    if ! wav-to-duration --read-entire-file=$read_entire_file scp:$data/wav.scp ark,t:$data/utt2dur 2>&1 | grep -v 'nonzero return status'; then
      echo "$0: there was a problem getting the durations; moving $data/utt2dur to $data/.backup/"
      mkdir -p $data/.backup/
      mv $data/utt2dur $data/.backup/
    fi
  fi
elif [ -f $data/feats.scp ]; then
  echo "$0: wave file does not exist so getting durations from feats files"
  feat-to-len scp:$data/feats.scp ark,t:- | awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' >$data/utt2dur
else
  echo "$0: Expected $data/wav.scp, $data/segments or $data/feats.scp to exist"
  exit 1
fi

len1=$(cat $data/utt2spk | wc -l)
len2=$(cat $data/utt2dur | wc -l)
if [ "$len1" != "$len2" ]; then
  echo "$0: warning: length of utt2dur does not equal that of utt2spk, $len2 != $len1"
  if [ $len1 -gt $[$len2*2] ]; then
    echo "$0: less than half of utterances got a duration: failing."
    exit 1
  fi
fi

echo "$0: computed $data/utt2dur"

exit 0
