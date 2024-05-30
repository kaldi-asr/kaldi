#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# Path to Fisher transcripts LM interpolation (if not defined only AMI transcript LM is built),
case $(hostname -d) in
  fit.vutbr.cz) FISHER_TRANS=/mnt/matylda2/data/FISHER/fe_03_p1_tran ;; # BUT,
  clsp.jhu.edu) FISHER_TRANS=/export/corpora4/ami/fisher_trans/part1 ;; # JHU,
  cstr.ed.ac.uk) FISHER_TRANS=`pwd`/eddie_data/lm/data/fisher/part1 ;; # Edinburgh,
  *) echo "Please modify the script to add your loaction of the Fisher transcripts, or modify this script."; exit 1;;
esac
# Or select manually,
# FISHER_TRANS=...
# Note: if you don't have the Fisher data, you can remove the --fisher $FISHER_TRANS
# option to local/ami_train_lms.sh below, and remove the case statement above.

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
local/ami_text_prep.sh data/local/downloads

local/ami_prepare_dict.sh
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

local/ami_train_lms.sh --fisher $FISHER_TRANS data/local/annotations/train.txt data/local/annotations/dev.txt data/local/dict/lexicon.txt data/local/lm

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
prune-lm --threshold=1e-7 data/local/lm/$final_lm.gz /dev/stdout | gzip -c > data/local/lm/$LM.gz
utils/format_lm.sh data/lang data/local/lm/$LM.gz data/local/dict/lexicon.txt data/lang_$LM

echo "Done"
exit 0
