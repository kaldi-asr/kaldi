#!/bin/bash -u

. ./cmd.sh
. ./path.sh

ICSI_TRANS=/disks/data1/corpora/icsi_mr_transcr #where to find ICSI transcriptions [required]
FISHER_TRANS=/disks/data1/corpora/LDC2004T19/fe_03_p1_tran #where to find FISHER transcriptions [optional, for LM esimation]

. utils/parse_options.sh

if ! command -v ngram-count >/dev/null 2>&1 ; then
  echo "$0: Error: the SRILM is not available or compiled" >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_srilm.sh" >&2
  exit 1
fi

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

#prepare dictionary and language resources
local/icsi_prepare_dict.sh
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

#prepare annotations, note: dict is assumed to exist when this is called
local/icsi_text_prep.sh $ICSI_TRANS data/local/annotations

local/icsi_train_lms.sh --fisher $FISHER_TRANS data/local/annotations/train.txt data/local/annotations/dev.txt data/local/dict/lexicon.txt data/local/lm

final_lm=$(cat data/local/lm/final_lm)
LM=$final_lm.pr1-7
#prune-lm --threshold=1e-7 data/local/lm/$final_lm.gz /dev/stdout | gzip -c > data/local/lm/$LM.gz
ngram -prune-lowprobs -unk -lm data/local/lm/$final_lm.gz -write-lm data/local/lm/$LM.gz
utils/format_lm.sh data/lang data/local/lm/$LM.gz data/local/dict/lexicon.txt data/lang_$LM

echo "Done"
exit 0

