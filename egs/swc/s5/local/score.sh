#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey) 2012
# Copyright University of Edinburgh (Author: Pawel Swietojanski) 2014
# Apache 2.0

orig_args=
for x in "$@"; do orig_args="$orig_args '$x'"; done

# begin configuration section.  we include all the options that score_sclite.sh or
# score_basic.sh might need, or parse_options.sh will die.
cmd=run.pl
stage=0
min_lmwt=9 # unused,
max_lmwt=15 # unused,
asclite=true
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [options] <data-dir> <lang-dir|graph-dir> <decode-dir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --asclite (true/false)          # score with ascltie instead of sclite (overlapped speech)"
  exit 1;
fi

data=$1

mic=$(echo $data | awk -F '/' '{print $2}')
case $mic in
  ihm*)
    echo "Using sclite for IHM (close talk),"
    eval local/score_asclite.sh --asclite false $orig_args
  ;;
  sdm*)
    echo "Using asclite for overlapped speech SDM (single distant mic),"
    eval local/score_asclite.sh --asclite $asclite $orig_args
  ;;
  mdm*)
    echo "Using asclite for overlapped speech MDM (multiple distant mics),"
    eval local/score_asclite.sh --asclite $asclite $orig_args
  ;;
  *)
    echo "local/score.sh: no ihm/sdm/mdm directories found. AMI recipe assumes data/{ihm,sdm,mdm}/..."
    exit 1;
  ;;
esac
