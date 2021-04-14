#!/bin/bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi

# Config:
cmd=run.pl
nj=40
use_gss_multiarray=false
use_gss_multiarray_allmics=false
bss_iterations=5
context_samples=240000
. utils/parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: local/run_gss.sh [options] <rttm-file> <chime6-corpus> <session-id> <log-dir> <enhanced-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --bss_iterations 5                       # Number of EM iterations"
   echo "  --context_samples 160000                 # Left-right context in number of samples"
   echo "  --multiarray <configuration>             # Multiarray configuration"
   exit 1;
fi

rttm_file=$1
session_id=$2
log_dir=$3
enhanced_dir=$4

miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir/ ]; then
  echo "$miniconda_dir/ does not exist. Please run '../../../tools/extras/install_miniconda.sh'"
  exit 1
fi

# Get absolute paths (since we will run enhancement from pb_chime5 dir)
enhanced_dir=$(utils/make_absolute.sh $enhanced_dir) || \
  { echo "Could not make absolute '$enhanced_dir'" && exit 1; }
rttm_file=$(utils/make_absolute.sh $rttm_file) || \
  { echo "Could not make absolute '$rttm_file'" && exit 1; }

if $use_gss_multiarray_allmics; then
  multiarray=True
elif $use_gss_multiarray; then
  multiarray=outer_array_mics
else
  multiarray=False
fi

$cmd JOB=1:$nj $log_dir/log/enhance_${session_id}.JOB.log \
  cd pb_chime5/ '&&' \
  $miniconda_dir/bin/python -m pb_chime5.scripts.kaldi_run_rttm with \
    storage_dir=${enhanced_dir} \
    chime6_dir="cache/CHiME6" \
    database_rttm=${rttm_file} \
    activity_rttm=${rttm_file} \
    session_id=${session_id} \
    job_id=JOB \
    number_of_jobs=${nj} \
    context_samples=${context_samples} \
    bss_iterations=${bss_iterations} \
    multiarray=${multiarray} || exit 1
