#!/bin/bash

# Begin configuration section.
nj=20
cmd=run.pl
stage=0
true_rttm_filename=None
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
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ -f $KALDI_ROOT/tools/VB_diarization/VB_diarization.py ]; then
    echo "VB_diarization is installed so will use the script"
else
    echo "VB_diarization is not installed, Please install
          it using extras/install_diarization_VBHMM.sh in tools/"
    exit 1;
fi

export VB_HMM_ROOT=$(cd $KALDI_ROOT/tools/VB_diarization/; pwd -P)
export PYTHONPATH=$VB_HMM_ROOT
echo $PYTHONPATH

if [ $# != 5 ]; then
  echo "Usage: local/VB_resegmentation.sh <data_dir> <init_rttm_filename> <output_dir> <dubm_model> <ie_model>"
  echo "Variational Bayes Re-segmenatation"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # How to run jobs."
  echo "  --nj <num-jobs|20>                               # Number of parallel jobs to run."
  echo "  --true-rttm-filename <string|None>               # The true rttm label file"
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

  exit 1;
fi

data_dir=$1
init_rttm_filename=$2
output_dir=$3
dubm_model=$4
ie_model=$5

mkdir -p $output_dir/rttm

sdata=$data_dir/split$nj;
utils/split_data.sh $data_dir $nj || exit 1;

if [ $stage -le 0 ]; then
    $cmd JOB=1:$nj $output_dir/log/VB_resegmentation.JOB.log \
      local/VB_resegmentation.py --true-rttm-filename $true_rttm_filename --max-speakers $max_speakers \
        --max-iters $max_iters --downsample $downsample --alphaQInit $alphaQInit \
	--sparsityThr $sparsityThr --epsilon $epsilon --minDur $minDur \
	--loopProb $loopProb --statScale $statScale --llScale $llScale \
	--channel $channel --initialize $initialize \
        $sdata/JOB $init_rttm_filename $output_dir $dubm_model $ie_model || exit 1;
fi
