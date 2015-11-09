#!/bin/bash

# Copyright 2012-2014  Guoguo Chen
# Apache 2.0.

# Begin configuration section.  
nj=8
cmd=run.pl
beam=-1             # Beam for proxy FST, -1 means no prune
phone_beam=-1       # Beam for KxL2xE FST, -1 means no prune
nbest=-1            # Use top n best proxy keywords in proxy FST, -1 means all
                    # proxies
phone_nbest=50      # Use top n best phone sequences in KxL2xE, -1 means all
                    # phone sequences
confusion_matrix=   # If supplied, using corresponding E transducer
count_cutoff=1      # Minimal count to be considered in the confusion matrix;
                    # will ignore phone pairs that have count less than this.
pron_probs=false    # If true, then lexicon looks like:
                    # Word Prob Phone1 Phone2...
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
echo "$0 " "$@"
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  echo "Generate proxy keywords for IV/OOV keywords. Phone confusions will be"
  echo "used when generating the proxies if the confusion matrix is supplied."
  echo "If you are going to use the confusion matrix, please use the following"
  echo "format for the file \$confusion_matrix:"
  echo "  p1 p2 count1        // For substitution"
  echo "  p3 <eps> count2     // For deletion"
  echo "  <eps> p4 count3     // For insertion"
  echo ""
  echo "Proxies keywords are generated using:"
  echo "K x L2 x E x L1'"
  echo "where K is a keyword FST, L2 is a lexicon that contains pronunciations"
  echo "of keywords in K, E is an edit distance FST that contains the phone"
  echo "confusions and L1 is the original lexicon."
  echo ""
  echo "The script assumes that L1.lex, L2.lex, words.txt and keywords.txt have"
  echo "been prepared and stored in the directory <kws-data-dir>."
  echo ""
  echo "Usage: local/generate_example_kws.sh <kws-data-dir>"
  echo " e.g.: local/generate_example_kws.sh data/dev10h/kws_proxy/"
  exit 1;
fi

set -e 
set -o pipefail

kwsdatadir=$1

# Checks some files.
for f in $kwsdatadir/L1.lex $kwsdatadir/L2.lex \
  $kwsdatadir/words.txt $kwsdatadir/keywords.txt; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1
done

# Gets phone symbols
phone_start=2
if $pron_probs; then
  phone_start=3
fi

pron_probs_param="";
if $pron_probs; then
  pron_probs_param="--pron-probs";
fi

ndisambig=`utils/add_lex_disambig.pl \
  $pron_probs_param $kwsdatadir/L1.lex $kwsdatadir/L1_disambig.lex`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $kwsdatadir/disambig.txt

cat $kwsdatadir/L2.lex $kwsdatadir/L1.lex |\
  awk '{for(i='$phone_start'; i <= NF; i++) {print $i;}}' |\
  sort -u | sed '1i\<eps>' |\
  cat - $kwsdatadir/disambig.txt | awk 'BEGIN{x=0} {print $0"\t"x; x++;}' \
  > $kwsdatadir/phones.txt

# Compiles lexicon into FST
cat $kwsdatadir/L2.lex |\
  utils/make_lexicon_fst.pl $pron_probs_param - |\
  fstcompile --isymbols=$kwsdatadir/phones.txt \
  --osymbols=$kwsdatadir/words.txt - |\
  fstinvert | fstarcsort --sort_type=olabel > $kwsdatadir/L2.fst

phone_disambig_symbol=`grep \#0 $kwsdatadir/phones.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $kwsdatadir/words.txt | awk '{print $2}'`
phone_disambig_symbols=`grep \# $kwsdatadir/phones.txt |\
  awk '{print $2}' | tr "\n" " "`
word_disambig_symbols=`grep \# $kwsdatadir/words.txt |\
  awk '{print $2}' | tr "\n" " "`
cat $kwsdatadir/L1_disambig.lex |\
  utils/make_lexicon_fst.pl $pron_probs_param - |\
  fstcompile --isymbols=$kwsdatadir/phones.txt \
  --osymbols=$kwsdatadir/words.txt - |\
  fstaddselfloops "echo $phone_disambig_symbol |" \
  "echo $word_disambig_symbol |" |\
  fstdeterminize | fstrmsymbols "echo $phone_disambig_symbols|" |\
  fstrmsymbols --remove-from-output=true "echo $word_disambig_symbols|" |\
  fstarcsort --sort_type=ilabel > $kwsdatadir/L1.fst

# Compiles E.fst
confusion_matrix_param=""
if [ ! -z $confusion_matrix ]; then
  echo "$0: Using confusion matrix, normalizing"
  local/count_to_logprob.pl --cutoff $count_cutoff \
    $confusion_matrix $kwsdatadir/confusion.txt
  confusion_matrix_param="--confusion-matrix $kwsdatadir/confusion.txt"
fi
cat $kwsdatadir/phones.txt |\
  grep -v -E "<.*>" | grep -v "SIL" | awk '{print $1;}' |\
  local/build_edit_distance_fst.pl --boundary-off=true \
  $confusion_matrix_param - - |\
  fstcompile --isymbols=$kwsdatadir/phones.txt \
  --osymbols=$kwsdatadir/phones.txt - $kwsdatadir/E.fst

# Pre-composes L2 and E, for the sake of efficiency
fstcompose $kwsdatadir/L2.fst $kwsdatadir/E.fst |\
  fstarcsort --sort_type=ilabel > $kwsdatadir/L2xE.fst

keywords=$kwsdatadir/keywords.int
# Prepares for parallelization
cat $kwsdatadir/keywords.txt |\
  utils/sym2int.pl -f 2- $kwsdatadir/words.txt | sort -R > $keywords

nof_keywords=`cat $keywords|wc -l`
if [ $nj -gt $nof_keywords ]; then
  nj=$nof_keywords
  echo "$0: Too many number of jobs, using $nj instead"
fi

# Generates the proxy keywords
mkdir -p $kwsdatadir/split/log
$cmd JOB=1:$nj $kwsdatadir/split/log/proxy.JOB.log \
  split -n r/JOB/$nj $keywords \| \
  generate-proxy-keywords --verbose=1 \
  --proxy-beam=$beam --proxy-nbest=$nbest \
  --phone-beam=$phone_beam --phone-nbest=$phone_nbest \
  $kwsdatadir/L2xE.fst $kwsdatadir/L1.fst ark:- ark:$kwsdatadir/split/proxy.JOB.fsts

proxy_fsts=""
for j in `seq 1 $nj`; do
  proxy_fsts="$proxy_fsts $kwsdatadir/split/proxy.$j.fsts"
done
cat $proxy_fsts > $kwsdatadir/keywords.fsts
