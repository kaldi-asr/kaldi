#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# Path to Fisher transcripts LM interpolation (if not defined only AMI transcript LM is built),
#FISHER_TRANS=`pwd`/eddie_data/lm/data/fisher/part1 # Edinburgh,
#FISHER_TRANS=/mnt/matylda2/data/FISHER/fe_03_p1_tran # BUT,
FISHER_TRANS=/export/ws15-ffs-data/corpora/LDC/LDC2004T19/fe_03_p1_tran # JSALT2015 workshop, cluster AWS-EC2,

# To run this script you need SRILM,
# JSALT2015 note : it's downloaded to /export/ws15-ffs-data/tools/srilm-1.7.1.tar.gz
! hash ngram-count && echo "Missing srilm, run 'cd ../../../tools/; ./install_srilm.sh" && exit 1

# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -x

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

