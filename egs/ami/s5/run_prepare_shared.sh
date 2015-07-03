#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# To run this script you need SRILM,

# Path to Fisher transcripts LM interpolation (if not defined only AMI transcript LM is built),
FISHER_TRANS=`pwd`/eddie_data/lm/data/fisher/part1 # Edinburgh, [DEFAULT]
# Path where AMI gets downloaded (or where locally available),
AMI_DIR=$PWD/DOWNLOAD/amicorpus # [DEFAULT]

# We can make setup specific to the 'domain' where the cluster is,
case "$(hostname -d)" in
  fit.vutbr.cz) # BUT cluster,
    FISHER_TRANS=/mnt/matylda2/data/FISHER/fe_03_p1_tran
    AMI_DIR=$(mktemp -d $(find /mnt/scratch*/$USER -maxdepth 0)/kaldi_ami_data_XXXXXX)
  ;;
  *) echo "Using defaults locations,"
  ;;
esac

# We can override the automatic setup by : 
# './run_prepare_shared.sh --AMI-DIR [dir] --FISHER-TRANS [dir]'
. utils/parse_options.sh 

# Load previous / store the new AMI_DIR location,
[ -r conf/ami_dir ] && AMI_DIR=$(cat conf/ami_dir) || echo $AMI_DIR >conf/ami_dir 

# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -x

local/ami_text_prep.sh $AMI_DIR

local/ami_prepare_dict.sh
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

local/ami_train_lms.sh --fisher $FISHER_TRANS data/local/annotations/train.txt data/local/annotations/dev.txt data/local/dict/lexicon.txt data/local/lm

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
prune-lm --threshold=1e-7 data/local/lm/$final_lm.gz /dev/stdout | gzip -c > data/local/lm/$LM.gz
utils/format_lm.sh data/lang data/local/lm/$LM.gz data/local/dict/lexicon.txt data/lang_$LM

echo "Done!"
exit 0

