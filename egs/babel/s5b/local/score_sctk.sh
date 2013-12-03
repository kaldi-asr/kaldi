#!/bin/bash
# Copyright Johns Hopkins University (Authors: Daniel Povey, Sanjeev Khudanpur) 2012-2013.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
cer=0
decode_mbr=true
min_lmwt=7
max_lmwt=17
model=
beam=7
word_ins_penalty=0.5
#end configuration section.

echo "$0 $@"

set -e 
set -o pipefail
set -u

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # (createCTM | filterCTM | runSclite)."
  echo "    --cer (0|1)                     # compute CER in addition to WER"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

name=`basename $data`; # e.g. eval2000

if [ $stage -le 1  ] ; then
  if [ -d data/local/w2s_extended ]; then  # this is for the syllable-based system.
    local/lattice_to_ctm_syllable.sh --decode-mbr $decode_mbr --min-lmwt $min_lmwt \
      --max-lmwt $max_lmwt --cmd "$cmd" --beam $beam --stage $stage \
      --word-ins-penalty $word_ins_penalty  $data $lang data/local/w2s_extended $dir 
  else
    local/lattice_to_ctm.sh --decode-mbr $decode_mbr --min-lmwt $min_lmwt \
      --max-lmwt $max_lmwt --cmd "$cmd" --beam $beam --stage $stage \
      --word-ins-penalty $word_ins_penalty  $data $lang $dir 
  fi
fi

if [ $stage -le 2 ] ; then 
  local/score_stm.sh --cmd "${cmd}" --cer $cer --min-lmwt ${min_lmwt} \
    --max-lmwt ${max_lmwt}  $data $lang $dir
fi

echo "Finished scoring on" `date`
exit 0
