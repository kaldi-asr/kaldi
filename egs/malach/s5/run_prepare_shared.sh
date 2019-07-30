#!/bin/bash -u

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

if ! command -v prune-lm >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

if ! command -v ngram-count >/dev/null 2>&1 ; then
  echo "$0: Error: the SRILM is not available or compiled" >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_srilm.sh" >&2
  exit 1
fi

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# Download of annotations, pre-processing,
# malach_dir must be set to where you have downloaded the malach data
# (see README.txt for where to get the data)
#
# For example: 
# malach_dir=/speech7/picheny5_nb/new_malach/malach_eng_speech_recognition/data

malach_dir=dummy_directory

local/malach_text_prep.sh $malach_dir

local/malach_prepare_dict.sh $malach_dir

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang


local/malach_train_lms.sh data/local/annotations/train.txt data/local/annotations/dev.txt data/local/dict/lexicon.txt data/local/lm

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-9
prune-lm --threshold=1e-9 data/local/lm/$final_lm.gz /dev/stdout | gzip -c > data/local/lm/$LM.gz
utils/format_lm.sh data/lang data/local/lm/$LM.gz data/local/dict/lexicon.txt data/lang_$LM

echo "Done"
exit 0
