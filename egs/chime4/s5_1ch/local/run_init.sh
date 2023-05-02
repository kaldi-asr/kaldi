#!/usr/bin/env bash

# Copyright 2016 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Config:
nj=30
stage=0 # resume training with --stage=N
eval_flag=true # make it true when the evaluation data are released

. utils/parse_options.sh || exit 1;

# This script is made from the kaldi recipe of the 2nd CHiME Challenge Track 2
# made by Chao Weng

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <Chime4 root directory>\n\n" `basename $0`
  echo "Please specifies a CHiME4 root directory"
  echo "If you use scripts distributed in the CHiME4 package,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

# clean data
chime4_data=$1
wsj0_data=$chime4_data/data/WSJ0 # directory of WSJ0 in Chime4. You can also specify your WSJ0 corpus directory

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $stage -le 0 ]; then
  # process for clean speech and making LMs etc. from original WSJ0
  # note that training on clean data means original WSJ0 data only (no booth data)
  local/clean_wsj0_data_prep.sh $wsj0_data
  local/wsj_prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang
  local/clean_chime4_format_data.sh
fi

echo "`basename $0` Done."
