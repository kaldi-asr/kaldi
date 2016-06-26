#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -o pipefail

stage=0
cmd=run.pl
special_symbol="***"    # Special symbol to be aligned with the inserted or deleted words. Your sentences should not contain this symbol. 

. path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "This script computes the oracle paths for the lattices "
  echo "and stores human-readable word alignment of the oracle and "
  echo "the reference\n"
  echo "Usage: $0 <data> <lang> <dir>"
  echo " e.g.: $0 data/train_si284 data/lang exp/tri4_bad_utts"
  exit 1
fi

data=$1
lang=$2
bad_utts_dir=$3

for f in $bad_utts_dir/lats/lat.1.gz $lang/oov.int $lang/words.txt $data/text; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done
  
nj=`cat $bad_utts_dir/lats/num_jobs`
oov=`cat $lang/oov.int`

utils/split_data.sh --per-utt $data $nj

sdata=$data/split$nj
 
dir=$bad_utts_dir/lattice_oracle
mkdir -p $dir

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/get_oracle.JOB.log \
    lattice-oracle --write-lattices="ark:|gzip -c > $dir/lat.JOB.gz" \
    "ark:gunzip -c $bad_utts_dir/lats/lat.JOB.gz |" \
    "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|" \
    ark,t:- ark,t:$dir/edits.JOB.txt \| \
    utils/int2sym.pl -f 2- $lang/words.txt '>' $dir/aligned_ref.JOB.txt || exit 1;
fi

echo $nj > $dir/num_jobs

if [ $stage -le 2 ]; then
  if [ -f $dir/aligned_ref.1.txt ]; then
    # the awk commands below are to ensure that partially-written files don't confuse us.
    for x in $(seq $nj); do cat $dir/aligned_ref.$x.txt; done | awk '{if(NF>=1){print;}}' > $dir/aligned_ref.txt
  else
    echo "$0: warning: no file $dir/aligned_ref.1.txt, using previously concatenated file if present."
  fi

  # in case any utterances failed to align, get filtered copy of $data/text
  utils/filter_scp.pl $dir/aligned_ref.txt < $data/text  > $dir/text
  cat $dir/text | awk '{print $1, (NF-1);}' > $dir/length.txt
  
  mkdir -p $dir/analysis

  align-text --special-symbol="$special_symbol"  ark:$dir/text ark:$dir/aligned_ref.txt  ark,t:- | \
    utils/scoring/wer_per_utt_details.pl --special-symbol "***" > $dir/analysis/per_utt_details.txt

  awk '{if ($2 == "#csid") print $1" "($4+$5+$6)}' $dir/analysis/per_utt_details.txt > $dir/edits.txt

  n1=$(wc -l < $dir/edits.txt)
  n2=$(wc -l < $dir/aligned_ref.txt)
  n3=$(wc -l < $dir/text)
  n4=$(wc -l < $dir/length.txt)
  if [ $n1 -ne $n2 ] || [ $n2 -ne $n3 ] || [ $n3 -ne $n4 ]; then
    echo "$0: mismatch in lengths of files:"
    wc $dir/edits.txt $dir/aligned_ref.txt $dir/text $dir/length.txt
    exit 1;
  fi

  # note: the format of all_info.txt is:
  # <utterance-id>   <number of errors>  <reference-length>  <decoded-output>   <reference>
  # with the fields separated by tabs, e.g.
  # adg04_sr009_trn 1 	12	 SHOW THE GRIDLEY+S TRACK IN BRIGHT ORANGE WITH HORNE+S IN DIM RED AT	 SHOW THE GRIDLEY+S TRACK IN BRIGHT ORANGE WITH HORNE+S IN DIM RED

  paste $dir/edits.txt \
      <(awk '{print $2}' $dir/length.txt) \
      <(awk '{$1="";print;}' <$dir/aligned_ref.txt) \
      <(awk '{$1="";print;}' <$dir/text) > $dir/all_info.txt

  sort -nr -k2 $dir/all_info.txt > $dir/all_info.sorted.txt

  if $cleanup; then
    rm $dir/edits.*.txt $dir/aligned_ref.*.txt || true
  fi
fi

if [ $stage -le 3 ]; then
  ###
  # These stats might help people figure out what is wrong with the data
  # a)human-friendly and machine-parsable alignment in the file per_utt_details.txt
  # b)evaluation of per-speaker performance to possibly find speakers with
  #   distinctive accents/speech disorders and similar
  # c)Global analysis on (Ins/Del/Sub) operation, which might be used to figure
  #   out if there is systematic issue with lexicon, pronunciation or phonetic confusability

  cat $dir/analysis/per_utt_details.txt | \
    utils/scoring/wer_per_spk_details.pl $data/utt2spk > $dir/analysis/per_spk_details.txt

  cat $dir/analysis/per_utt_details.txt | \
    utils/scoring/wer_ops_details.pl --special-symbol "$special_symbol" | \
    sort -i -b -k1,1 -k4,4nr -k2,2 -k3,3 > $dir/analysis/ops_details.txt
fi

