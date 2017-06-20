#!/bin/bash
# Copyright 2013-2014  Johns Hopkins University (authors: Yenda Trmal, Daniel Povey)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

#Simple utility script to convert the gzipped ARPA lm into a G.fst file


oov_prob_file=
unk_fraction=
cleanup=true
#end configuration section.



echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <arpa-lm-file> <lang-dir> <dest-dir>"
  echo "Options: --oov-prob-file <oov-prob-file>   # e.g. data/local/oov2prob"
  echo "           # with this option it will replace <unk> with OOVs in G.fst."
  exit 1;
fi

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code

lmfile=$1
langdir=$2
destdir=$3

mkdir $destdir 2>/dev/null || true


if [ ! -z "$oov_prob_file" ]; then
  if [ ! -s "$oov_prob_file" ]; then
    echo "$0: oov-prob file $oov_prob_file does not exist"
    exit 1;
  fi
  if [ -z "$unk_fraction" ]; then
    echo "--oov-prob option requires --unk-fraction option";
    exit 1;
  fi

  min_prob=$(gunzip -c $lmfile | perl -e '  $minlogprob = 0.0;
     while(<STDIN>) { if (m/\\(\d)-grams:/) { $order = $1; }
      if ($order == 1) { @A = split;
       if ($A[0] < $minlogprob && $A[0] != -99) { $minlogprob = $A[0]; }}} print $minlogprob')
  echo "Minimum prob in LM file is $min_prob"

  echo "$0: creating LM file with unk words, using $oov_prob_file, in $destdir/lm_tmp.gz"
  gunzip -c $lmfile | \
    perl -e ' ($oov_prob_file,$min_prob,$unk_fraction) = @ARGV; $ceilinged=0;
      $min_prob < 0.0 || die "Bad min_prob"; # this is a log-prob
      $unk_fraction > 0.0 || die "Bad unk_fraction"; # this is a prob
      open(F, "<$oov_prob_file") || die "opening oov file";
      while (<F>) { push @OOVS, $_; }
      $num_oovs = @F;
      while(<STDIN>) {
      if (m/^ngram 1=(\d+)/) { $n = $1 + $num_oovs; print "ngram 1=$n\n"; }
      else { print; } # print all lines unchanged except the one that says ngram 1=X.
      if (m/^\\1-grams:$/) {
        foreach $l (@OOVS) {
          @A = split(" ", $l);
          @A == 2 || die "bad line in oov2prob: $_;";
          ($word, $prob) = @A;
          $log10prob = (log($prob * $unk_fraction) / log(10.0));
          if ($log10prob > $min_prob) { $log10prob = $min_prob; $ceilinged++;}
          print "$log10prob $word\n";
       }
     }} print STDERR "Ceilinged $ceilinged unk-probs\n";' \
       $oov_prob_file $min_prob $unk_fraction | gzip -c > $destdir/lm_tmp.gz
  lmfile=$destdir/lm_tmp.gz
fi

if [[ $lmfile == *.bz2 ]] ; then
  decompress="bunzip2 -c $lmfile"
elif [[ $lmfile == *.gz ]] ; then
  decompress="gunzip -c $lmfile"
else
  decompress="cat $lmfile"
fi

$decompress | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$langdir/words.txt - $destdir/G.fst || exit 1

fstisstochastic $destdir/G.fst || true;

if $cleanup; then
  rm $destdir/lm_tmp.gz  2>/dev/null || true;
fi

exit 0
