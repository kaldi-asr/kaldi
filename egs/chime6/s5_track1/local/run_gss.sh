#!/bin/bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi

# Config:
cmd=run.pl
nj=4
use_multiarray=false

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/run_gss.sh [options] <session-id> <log-dir> <enhanced-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --bmf \"1 2 3 4\"                        # microphones used for beamforming"
   exit 1;
fi

session_id=$1
log_dir=$2
enhanced_dir=$3

if [ ! -d pb_chime5/ ]; then
  echo "Missing pb_chime5, run 'local/install_pb_chime5'" 
  exit 1
fi

miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir/ ]; then
  echo "$miniconda_dir/ does not exist. Please run '../../../tools/extras/install_miniconda.sh'"
  exit 1
fi

enhanced_dir=$(utils/make_absolute.sh $enhanced_dir) || \
  { echo "Could not make absolute '$enhanced_dir'" && exit 1; }

if $use_multiarray; then
  multiarray=True
else
  multiarray=False
fi

$cmd JOB=1:$nj $log_dir/log/enhance_${session_id}.JOB.log \
  cd pb_chime5/ '&&' \
  $miniconda_dir/bin/python -m pb_chime5.scripts.kaldi_run with \
    storage_dir=$enhanced_dir \
    session_id=$session_id \
    job_id=JOB number_of_jobs=$nj \
    multiarray=$multiarray || exit 1
