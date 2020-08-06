#!/bin/bash

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

# References:
#  - https://github.com/srvk/eesen/blob/master/asr_egs/wsj/utils/ctc_compile_dict_token.sh
#
#  - EESEN: End-to-End Speech Recognition using Deep RNN Models and
#      WFST-based Decoding (https://arxiv.org/pdf/1507.08240.pdf)

set -e

echo "$0 $@"  # Print the command line for logging

if [ $# != 3 ]; then
  echo "usage: $0 <lexicon_txt> <lm-gz> <output-lang-dir>"
  exit 1
fi

. ./cmd.sh
. ./path.sh

dict=$1
lm=$2
dir=$3

[ ! -f $dict ] && echo "$dict does not exit!" && exit 1
[ ! -f $lm ] && echo "$lm does not exit!" && exit 1

mkdir -p $dir

cp $dict $dir/lexicon.txt

cat $dir/lexicon.txt | cut -d ' ' -f2- | tr -s ' ' '\n' | sort | uniq > $dir/phones.list

perl -ape 's/(\S+\s+)(.+)/${1}1.0 $2/;' < $dir/lexicon.txt > $dir/lexiconp.txt || exit 1

ndisambig=$(utils/add_lex_disambig.pl $dir/lexiconp.txt $dir/lexiconp_disambig.txt)
ndisambig=$[$ndisambig+1]

for ((i=0; i<=$ndisambig; i++)); do
  echo '#'$i
done > $dir/disambig.list

(
  echo '<eps>'
  echo '<blk>'
) | cat - $dir/phones.list $dir/disambig.list | awk '{print $1, NR-1}' > $dir/tokens.txt

if [[ ! -f $dir/T.fst ]]; then
  local/token_to_fst.py --tokens-txt-filename $dir/tokens.txt |
    fstcompile \
      --isymbols=$dir/tokens.txt \
      --osymbols=$dir/tokens.txt \
      --keep_isymbols=false \
      --keep_osymbols=false |
    fstarcsort --sort_type=olabel > $dir/T.fst || exit 1
fi

cat $dir/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR + 1);
  }' > $dir/words.txt || exit 1


token_disambig_symbol=$(grep \#0 $dir/tokens.txt | awk '{print $2}')
word_disambig_symbol=$(grep \#0 $dir/words.txt | awk '{print $2}')

silprob=0
silphone="sil"

if [[ ! -f $dir/L.fst ]]; then
  utils/make_lexicon_fst.pl \
      --pron-probs $dir/lexiconp_disambig.txt $silprob $silphone '#'$ndisambig |
    fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/words.txt \
      --keep_isymbols=false --keep_osymbols=false |
    fstaddselfloops "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" |
    fstarcsort --sort_type=olabel > $dir/L.fst || exit 1
fi


if [[ ! -f $dir/G.fst ]]; then
  gunzip -c $lm |
    grep -v '<s> <s>' |
    grep -v '</s> <s>' |
    grep -v '</s> </s>' |
    arpa2fst - |
    fstprint |
    utils/eps2disambig.pl |
    utils/s2eps.pl |
    fstcompile \
      --isymbols=$dir/words.txt \
      --osymbols=$dir/words.txt  \
      --keep_isymbols=false \
      --keep_osymbols=false |
    fstrmepsilon |
    fstarcsort --sort_type=ilabel > $dir/G.fst
fi

set +e
fstisstochastic $dir/G.fst
set -e

# The output is like:
# 9.14233e-05 -0.259833
# we do expect the first of these 2 numbers to be close to zero (the second is
# nonzero because the backoff weights make the states sum to >1).

if true; then
  # Everything in this "if" statement is only for diagnostic.
  # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
  # this might cause determinization failure of CLG.
  # #0 is treated as an empty word.
  mkdir -p $dir/tmpdir.g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }}
       END{print "0 0 #0 #0"; print "0";}' \
       < "$dir/lexicon.txt" > $dir/tmpdir.g/select_empty.fst.txt

  fstcompile --isymbols=$dir/words.txt --osymbols=$dir/words.txt \
    $dir/tmpdir.g/select_empty.fst.txt \
    | fstarcsort --sort_type=olabel \
    | fstcompose - $dir/G.fst > $dir/tmpdir.g/empty_words.fst

  fstinfo $dir/tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' \
    && echo "Language model has cycles with empty words" && exit 1

  rm -r $dir/tmpdir.g
fi

fsttablecompose $dir/L.fst $dir/G.fst |
  fstdeterminizestar --use-log=true |
  fstminimizeencoded |
  fstarcsort --sort_type=ilabel > $dir/LG.fst || exit 1

set +e
fstisstochastic $dir/LG.fst
set -e

fsttablecompose $dir/T.fst $dir/LG.fst > $dir/TLG.fst || exit 1

fstconvert --fst_type=const TLG.fst const_TLG.fst
mv const_TLG.fst TLG.fst

set +e
fstisstochastic $dir/TLG.fst
set -e

# remove files not needed any more
for f in G.fst L.fst T.fst LG.fst disambig.list \
         lexiconp.txt lexiconp_disambig.txt; do
  rm $dir/$f
done
