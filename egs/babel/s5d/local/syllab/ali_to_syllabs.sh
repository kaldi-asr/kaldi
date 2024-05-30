#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
cmd=run.pl
# End configuration section
. ./utils/parse_options.sh

if [ -f ./path.sh ]; then . ./path.sh; fi

if [ $# != 4 ]; then
  echo "This script takes an ali directory and syllab lang dir and generates"
  echo "syllabic transceription of the alignment"
  echo ""
  echo "Usage: $0 <data-dir> <syllab-lang-dir> <ali-dir> <out-dir>"
  echo " e.g.: $0 data/train data/lang_syll exp/tri5_ali exp/tri5_ali_syll"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) "

  exit 1;
fi

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


data=$1
lang=$2
ali=$3
out=$4


for f in real_words.txt lex.words2syllabs.fst ; do
  [ ! -f $lang/$f ] && \
    echo "The given lang directory is probably not a syllable lang dir" && \
    echo "The file $lang/$f is missing" && \
    exit 1
done

for f in words.txt L.fst ; do
  [ ! -f $lang/$f ] && \
    echo "The given lang directory does not contain the $f file" && \
    exit 1
done

for f in $ali/num_jobs $ali/final.mdl $ali/ali.1.gz  ; do
  [ ! -f $f ] && \
    echo "The given lang directory does not contain the $f file" && \
    exit 1
done

nj=$(cat $ali/num_jobs)
echo "Extracting phoneme sequences"
$cmd JOB=1:$nj $out/log/ali-to-phones.JOB.log \
  ali-to-phones $ali/final.mdl ark:"gunzip -c $ali/ali.JOB.gz|" ark:- \| \
    transcripts-to-fsts ark:-  ark:$out/phones.JOB.fst || exit 1

echo "Composing with files in $lang to get syllable sequences"
$cmd JOB=1:$nj $out/log/get-syll-text.JOB.log \
  cat $data/split$nj/JOB/text \| sym2int.pl -f 2- --map-oov '\<unk\>' $lang/real_words.txt \| \
    transcripts-to-fsts ark,t:- ark:- \|\
    fsttablecompose $lang/lex.words2syllabs.fst ark:- ark:-\| \
    fsts-project ark:- ark:-\| \
    fsttablecompose $lang/L.fst ark:- ark:- \|\
    fsttablecompose ark:$out/phones.JOB.fst ark:- ark:- \| \
    fsts-to-transcripts ark:- ark,t:"|int2sym.pl -f 2- $lang/words.txt > $out/text.JOB"
cat $out/text.* | sort > $out/text

echo "Done"

