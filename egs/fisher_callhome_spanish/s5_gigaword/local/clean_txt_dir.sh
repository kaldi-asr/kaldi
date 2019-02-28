#!/bin/bash

# Script to clean up gigaword LM text
# Removes punctuations, does case normalization

stage=0
nj=500

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <textdir> <outdir>"
    exit 1;
fi

txtdir=$1
textdir=$(realpath $txtdir)
outdir=$(realpath $2)

workdir=$outdir/tmp
if [ $stage -le 0 ]; then
  rm -rf $outdir
  mkdir -p $workdir
  mkdir -p $textdir/splits
  mkdir -p $outdir/data
  split -l 1000000 $textdir/in.txt $textdir/splits/out
  numsplits=0
  for x in $textdir/splits/*; do
    numsplits=$((numsplits+1))
    ln -s $x $outdir/data/$numsplits
  done
  echo $numsplits
  cp $SPARROWHAWK_ROOT/documentation/grammars/sentence_boundary_exceptions.txt .
  $train_cmd --max_jobs_run 100 JOB=1:$numsplits $outdir/sparrowhawk/log/JOB.log \
    local/run_norm.sh \
    sparrowhawk_configuration.ascii_proto \
    $SPARROWHAWK_ROOT/language-resources/esp/sparrowhawk/ \
    $outdir/data \
    JOB \
    $outdir/sparrowhawk/
  cat $outdir/sparrowhawk/*.txt | sed "/^$/d"  > $outdir/text_normalized

  # check if numbers are there in normalized output
  awk '{for(i=1;i<=NF;i++) {if (!seen[$i]) {print $i; seen[$i]=1} }}' \
    $outdir/text_normalized > $outdir/unique_words
  grep "[0-9]" $outdir/unique_words | sort -u >  $outdir/numbers
fi
