#!/bin/bash
# Apache 2.0


#[ -f ./path.sh ] && . ./path.sh
#. parse_options.sh || exit 1;

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  echo "Usage: local/add_sclite.sh <score-dir> <words.txt> <ref-trn> [<ref-clevel-trn>]"
  echo "  sclite evaluation on word and phoneme level for all *.tra files in score-dir"
  exit 1;
fi

oldLC=$LC_ALL
export LC_ALL="en_US.utf8"

scoreDir=$1
wordsTxt=$2
refWordTrn=$3
refPhoneTrn=$4

for traFile in $(ls $scoreDir/*.tra)
do
  ./utils/int2sym.pl -f 2- $wordsTxt $traFile | awk '{print $0" ("$1")"}' | cut -d' ' -f2- >  $traFile.trn
  cat $traFile.trn | ./local/word2char-trn.sh > $traFile.clevel.trn
  ../../tools/sctk/bin/sclite -r $refWordTrn trn -h $traFile.trn trn -o sum pralign -i rm -s 
  if [ -n "$refPhoneTrn" ]; then
     ../../tools/sctk/bin/sclite -r $refPhoneTrn trn -h $traFile.clevel.trn trn -o sum pralign -i rm -s 
  fi
done

export LC_ALL=$oldLC

exit 0;
