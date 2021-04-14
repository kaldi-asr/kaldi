#!/usr/bin/env bash

# Copyright 2019  Zili Huang

# This script is a wrapper for Variational Bayes resegmentation.
# It shows how to use the code from Brno University of Technology 
# to do resegmentation.

# Begin configuration section.
nj=20
cmd=run.pl
stage=0
max_speakers=10
max_iters=10
downsample=25
alphaQInit=100.0
sparsityThr=0.001
epsilon=1e-6
minDur=1
loopProb=0.9
statScale=0.2
llScale=1.0
channel=0
initialize=1

# The following performs VB-based overlap assignment (see https://arxiv.org/abs/1910.11646)
# It should be an RTTM file marking the segments which contain overlaps.
# See, for example, egs/chime6/s5b_track2/local/overlap/detect_overlaps.sh for
# using Pyannote overlap detector to get such an RTTM.
overlap_rttm=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: diarization/VB_resegmentation.sh <data_dir> <init_rttm_filename> <output_dir> <dubm_model> <ie_model>"
  echo "Variational Bayes Re-segmenatation"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # How to run jobs."
  echo "  --nj <num-jobs|20>                               # Number of parallel jobs to run."
  echo "  --max-speakers <n|10>                            # Maximum number of speakers" 
  echo "                                                   # expected in the utterance" 
  echo "					           # (default: 10)"
  echo "  --max-iters <n|10>                               # Maximum number of algorithm"
  echo "                                                   # iterations (default: 10)" 
  echo "  --downsample <n|25>                              # Perform diarization on input"
  echo "                                                   # downsampled by this factor"
  echo "                                                   # (default: 25)"
  echo "  --alphaQInit <float|100.0>                       # Dirichlet concentraion"
  echo "                                                   # parameter for initializing q"
  echo "  --sparsityThr <float|0.001>                      # Set occupations smaller that"
  echo "                                                   # this threshold to 0.0 (saves"
  echo "                                                   # memory as the posteriors are"
  echo "                                                   # represented by sparse matrix)"
  echo "  --epsilon <float|1e-6>                           # Stop iterating, if obj. fun." 
  echo "                                                   # improvement is less than" 
  echo "				                   # epsilon"
  echo "  --minDur <n|1>                                   # Minimum number of frames"
  echo "                                                   # between speaker turns imposed"
  echo "                                                   # by linear chains of HMM" 
  echo "                                                   # state corresponding to each" 
  echo "                                                   # speaker. All the states in"
  echo "                                                   # a chain share the same output"
  echo "                                                   # distribution"
  echo "  --loopProb <float|0.9>                           # Probability of not switching"
  echo "                                                   # speakers between frames"
  echo "  --statScale <float|0.2>                          # Scale sufficient statistics" 
  echo "                                                   # collected using UBM"
  echo "  --llScale <float|1.0>                            # Scale UBM likelihood (i.e."
  echo "                                                   # llScale < 1.0 make" 
  echo "                                                   # attribution of frames to UBM"
  echo "                                                   # componets more uncertain)" 
  echo "  --channel <n|0>                                  # Channel information in the rttm file"
  echo "  --initialize <n|1>                               # Whether to initalize the"
  echo "                                                   # speaker posterior (if not)"
  echo "                                                   # the speaker posterior will be"
  echo "                                                   # randomly initilized"
  echo "  --overlap-rttm <str|>                            # If provided, performs overlap"
  echo "                                                   # assignment using Q-matrix"

  exit 1;
fi

data_dir=$1
init_rttm_filename=$2
output_dir=$3
dubm_model=$4
ie_model=$5

mkdir -p $output_dir/tmp

sdata=$data_dir/split$nj;
utils/split_data.sh $data_dir $nj || exit 1;

save_opts=""
if [ ! -z "$overlap_rttm" ]; then
  save_opts="--save-posterior"
fi

if [ $stage -le 0 ]; then
  # Dump the diagonal UBM model into txt format.
  $cmd $output_dir/log/convert_diag_ubm.log \
    gmm-global-copy --binary=false \
      $dubm_model \
      $output_dir/tmp/dubm.tmp || exit 1;

  # Dump the ivector extractor model into txt format.
  $cmd $output_dir/log/convert_ie.log \
    ivector-extractor-copy --binary=false \
      $ie_model \
      $output_dir/tmp/ie.tmp || exit 1;
fi

if [ $stage -le 1 ]; then
  # VB resegmentation
  $cmd JOB=1:$nj $output_dir/log/VB_resegmentation.JOB.log \
    python3 diarization/VB_resegmentation.py --max-speakers $max_speakers \
      --max-iters $max_iters --downsample $downsample --alphaQInit $alphaQInit \
      --sparsityThr $sparsityThr --epsilon $epsilon --minDur $minDur \
      --loopProb $loopProb --statScale $statScale --llScale $llScale \
      --channel $channel --initialize $initialize "$save_opts" \
      $sdata/JOB $init_rttm_filename $output_dir/tmp $output_dir/tmp/dubm.tmp $output_dir/tmp/ie.tmp || exit 1;

  cat $output_dir/tmp/*_predict.rttm > $output_dir/VB_rttm
fi

if [ $stage -le 2 ] && [ ! -z "$overlap_rttm" ]; then
  # Overlap assignment
  $cmd JOB=1:$nj $output_dir/log/VB_overlap.JOB.log \
    python3 diarization/VB_overlap_assign.py \
      $sdata/JOB $overlap_rttm $init_rttm_filename $output_dir/tmp || exit 1;

  cat $output_dir/tmp/*ovl*.rttm > $output_dir/VB_rttm_ol
fi
