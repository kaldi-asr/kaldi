#!/usr/bin/env bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi

# Config:
cmd=run.pl
nj=4
multiarray=outer_array_mics
bss_iterations=5
context_samples=160000
. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/run_gss.sh [options] <session-id> <log-dir> <enhanced-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --bss_iterations 5                       # Number of EM iterations"
   echo "  --context_samples 160000                 # Left-right context in number of samples"
   echo "  --multiarray <configuration>             # Multiarray configuration"
   exit 1;
fi

# setting multiarray as "true" uses all mics, we didn't see any performance
# gain from this we have chosen settings that makes the enhacement finish
# in around 1/3 of a day without significant change in performance.
# our result during the experiments are as follows:

#MAF: multi array = False
#MAT: multi array = True
#Enhancement  Iterations  Num Microphones  Context  Computational time for GSS  #cpus  dev WER  eval WER
#GSS(MAF)     10           24                        17   hrs                   30     62.3     57.98
#GSS(MAT)      5           24               10s      26   hrs                   50     53.15    53.77
#GSS(MAT)      5           12               10s       9.5 hrs                   50     53.09    53.75

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

$cmd JOB=1:$nj $log_dir/log/enhance_${session_id}.JOB.log \
  cd pb_chime5/ '&&' \
  $miniconda_dir/bin/python -m pb_chime5.scripts.kaldi_run with \
    chime6=True \
    storage_dir=$enhanced_dir \
    session_id=$session_id \
    job_id=JOB number_of_jobs=$nj \
    bss_iterations=$bss_iterations \
    context_samples=$context_samples \
    multiarray=$multiarray || exit 1
