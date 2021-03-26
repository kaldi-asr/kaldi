#!/usr/bin/env bash

set -e
cmd=run.pl
stage=0
decode_mbr=false
stats=true
beam=6
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
iter=final

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

decode_dir=$3
steps/scoring/score_kaldi_wer.sh --word_ins_penalty $word_ins_penalty \
  --min_lmwt $min_lmwt --max_lmwt $max_lmwt "$@"

steps/scoring/score_kaldi_cer.sh --word_ins_penalty $word_ins_penalty \
  --min_lmwt $min_lmwt --max_lmwt $max_lmwt --stage 2 "$@"

local/score_paragraph.sh --word_ins_penalty $word_ins_penalty \
  --min_lmwt $min_lmwt --max_lmwt $max_lmwt $decode_dir
