#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This script prepares speech labels for 
# training unsad network for speech activity detection and music detection.
# This is similar to the script prepare_unsad_data.sh, but directly 
# uses existing alignments to create labels, instead of creating new alignments.

set -e
set -o pipefail
set -u

. path.sh

stage=-2
cmd=queue.pl

# Options to be passed to get_sad_map.py 
map_noise_to_sil=true   # Map noise phones to silence label (0)
map_unk_to_speech=true  # Map unk phones to speech label (1)
sad_map=    # Initial mapping from phones to speech/non-speech labels.
            # Overrides the default mapping using phones/silence.txt 
            # and phones/nonsilence.txt
speed_perturb=false

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "This script takes a data directory and alignment directory and "
  echo "converts it into speech activity labels"
  echo "for the purpose of training a Universal Speech Activity Detector.\n"
  echo "Usage: $0 [options] <data-dir> <lang> <ali-dir> <temp-dir>"
  echo " e.g.: $0 data/train_100k data/lang exp/tri4a_ali exp/vad_data_prep"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (run.pl|/queue.pl <queue opts>)            # how to run jobs."
  exit 1
fi

data_dir=$1
lang=$2
ali_dir=$3
dir=$4

extra_files=

for f in $lang/phones.txt $lang/phones/silence.txt $lang/phones/nonsilence.txt $sad_map $ali_dir/ali.1.gz $ali_dir/final.mdl $ali_dir/tree $extra_files; do
  if [ ! -f $f ]; then
    echo "$f could not be found"
    exit 1
  fi
done

mkdir -p $dir

data_id=$(basename $data_dir)

if [ $stage -le 0 ]; then
  # Get a mapping from the phones to the speech / non-speech labels
  steps/segmentation/get_sad_map.py \
    --init-sad-map="$sad_map" \
    --map-noise-to-sil=$map_noise_to_sil \
    --map-unk-to-speech=$map_unk_to_speech \
    $lang | utils/sym2int.pl -f 1 $lang/phones.txt > $dir/sad_map
fi

###############################################################################
# Convert alignment into SAD labels at utterance-level in segmentation format
###############################################################################

vad_dir=$dir/`basename ${ali_dir}`_vad_${data_id}

# Convert relative path to full path
vad_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir;' $vad_dir ${PWD}`

if [ $stage -le 1 ]; then
  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$cmd" \
    $ali_dir $dir/sad_map $vad_dir
fi

[ ! -s $vad_dir/sad_seg.scp ] && echo "$0: $vad_dir/sad_seg.scp is empty" && exit 1

###############################################################################
# Post-process the segmentation and create frame-level alignments and 
# per-frame deriv weights.
###############################################################################

if [ $stage -le 2 ]; then
  # Create per-frame speech / non-speech labels. 
  nj=`cat $vad_dir/num_jobs`
  
  set +e 
  for n in `seq $nj`; do
    utils/create_data_link.pl $vad_dir/speech_labels.$n.ark
  done
  set -e

  utils/data/get_utt2num_frames.sh $data_dir
  if ! $speed_perturb; then
    $cmd JOB=1:$nj $vad_dir/log/get_speech_labels.JOB.log \
      segmentation-copy --keep-label=1 scp:$vad_dir/sad_seg.JOB.scp ark:- \| \
      segmentation-to-ali --lengths-rspecifier=ark,t:$data_dir/utt2num_frames \
        ark:- ark,scp:$vad_dir/speech_labels.JOB.ark,$vad_dir/speech_labels.JOB.scp
  else
    awk '{print $1" "$2; print "sp0.9-"$1" "int($2 / 0.9); print "sp1.1-"$1" "int($2 / 1.1)}' $data_dir/utt2num_frames > \
      $vad_dir/utt2num_frames_sp
    $cmd JOB=1:$nj $vad_dir/log/get_speech_labels.JOB.log \
      segmentation-copy --keep-label=1 scp:$vad_dir/sad_seg.JOB.scp ark:- \| \
      segmentation-speed-perturb --speeds=0.9:1.0:1.1 ark:- ark:- \| \
      segmentation-to-ali --lengths-rspecifier=ark,t:$vad_dir/utt2num_frames_sp \
        ark:- ark,scp:$vad_dir/speech_labels.JOB.ark,$vad_dir/speech_labels.JOB.scp
  fi

  for n in `seq $nj`; do
    cat $vad_dir/speech_labels.$n.scp
  done | sort -k1,1 > $vad_dir/speech_labels.scp

  cp $vad_dir/speech_labels.scp $data_dir
fi

echo "$0: Finished creating corpus for training Universal SAD with data in $data_dir and labels in $vad_dir"
