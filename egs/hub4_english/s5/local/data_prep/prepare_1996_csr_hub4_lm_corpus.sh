#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares the 1996 CSR HUB4 Language Model corpus
# https://catalog.ldc.upenn.edu/LDC98T31

set -e
set -o pipefail
set -u

nj=4
cmd=run.pl
stage=0

[ -f ./path.sh ] && . ./path.sh

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo " e.g.: $0 /export/corpora/LDC/LDC98T31/1996_csr_hub4_model data/local/data/csr96_hub4"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

mkdir -p $dir

for d in $SOURCE_DIR/st_train/ $SOURCE_DIR/st_test/; do
  if [ ! -d $d ]; then
    echo "$0: Invalid SOURCE-DIR $SOURCE_DIR for LDC98T31 corpus"
    exit 1
  fi
  ls $d/*.stZ 
done | sort > $dir/filelist

mkdir -p $dir/split$nj/

if [ $stage -le 1 ]; then
  eval utils/split_scp.pl $dir/filelist $dir/split$nj/filelist.{`seq -s, $nj | sed 's/,$//'`}
  $cmd JOB=1:$nj $dir/log/process_text.JOB.log \
    local/data_prep/process_1996_csr_hub4_lm_filelist.py \
    $dir/split$nj/filelist.JOB $dir
fi

for x in `ls $SOURCE_DIR/st_train/*.stZ`; do
  y=`basename $x`
  name=${y%.stZ}
  echo $dir/${name}.txt.gz
done > $dir/train.filelist

for x in `ls $SOURCE_DIR/st_test/*.stZ`; do
  y=`basename $x`
  name=${y%.stZ}
  echo $dir/${name}.txt.gz
done > $dir/test.filelist
