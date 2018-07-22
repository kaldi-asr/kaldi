#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares the North American News Text Supplement Corpus
# https://catalog.ldc.upenn.edu/LDC98T30

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
  echo " e.g.: $0 /export/corpora/LDC/LDC98T30/northam_news_txt_sup data/local/data/na_news_supp"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

dir_list=

rm -f $dir/.error 2>/dev/null

for x in $SOURCE_DIR/nyt/*/ $SOURCE_DIR/latwp/ $SOURCE_DIR/apws/*/; do
  year=`basename $x`
  newspaper=`basename $(dirname $x)`

  d=$dir/${newspaper}_${year}
  
  if [ $year == latwp ]; then
    d=$dir/latwp_1997
  elif [ $year == english ]; then
    d=$dir/apws
  fi

  mkdir -p $d

  dir_list="$dir_list $d"

  list_file=$d/articles.list
  ls $x/*.gz > $list_file
  
  mkdir -p $d/split$nj

  eval utils/split_scp.pl $d/articles.list \
    $d/split$nj/articles.list.{`seq -s, $nj`}

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
