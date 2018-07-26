#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares the North American News Text Corpus
# https://catalog.ldc.upenn.edu/LDC95T21

[ -f ./path.sh ] && . ./path.sh
. ./cmd.sh

set -e
set -o pipefail
set -u

nj=4
cmd=run.pl

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <DIR>"
  echo " e.g.: $0 /export/corpora/LDC/LDC95T21 data/local/data/na_news"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

dir_list=

rm -f $dir/.error 2>/dev/null

for x in $SOURCE_DIR/*/*/*; do
  year=`basename $x`
  newspaper=`basename $(dirname $x)`
  d=$dir/${newspaper}_${year}

  dir_list="$dir_list $d"

  list_file=$d/articles.list
  ls $x/*.gz > $list_file
  
  mkdir -p $d/split$nj

  eval utils/split_scp.pl $d/articles.list \
    $d/split$nj/articles.list.{`seq -s, $nj | sed 's/,$//'`}

  $cmd JOB=1:$nj $d/log/get_processed_text.JOB.log \
    local/data_prep/process_na_news_text.py $d/split$nj/articles.list.JOB \
    $d/corpus.JOB.gz || touch $dir/.error &
done

wait

if [ -f $dir/.error ]; then
  echo "$0: Failed to process files."
fi

for d in $dir_list; do
  gunzip -c $d/corpus.*.gz | gzip -c > $d/corpus.gz || exit 1
  rm $d/corpus.*.gz
done
