#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

nj=64
cmd=run.pl
beam=5                    # Beam for proxy FST; usually used together with the nbest option
nbest=100                 # First n best proxy keywords
phone_cutoff=5            # We don't generate proxy keywords for OOV keywords that have less phones
                          # than the specified cutoff; they may introduce more false alarms
count_cutoff=1            # Cutoff for the phone confusion pair counts 
confusion_matrix=
pron_probs=false          # If true, then lexicon looks like:
                          # Word Prob Phone1 Phone2...
keyword_set="OOV"         # Generate proxies for either IV keywords or OOV keywords

[ -f ./path.sh ] && . ./path.sh; # source the path.
echo "$0 " "$@"
. parse_options.sh || exit 1;

set -e 
set -o pipefail
set -u
set -x

if [ $# -ne 5 ]; then
  echo "Generate proxy keywords for IV/OOV keywords. You may apply the confusion matrix."
  echo "If you are going to use the confusion matrix, please use the following format for"
  echo "the file \$confusion_matrix:"
  echo "          p1 p2 count1        // For substitution"
  echo "          p3 <eps> count2     // For deletion"
  echo "          <eps> p4 count3     // For insertion"
  echo ""
  echo "Proxies keywords are generated using:"
  echo "K x L2 x E x L1'"
  echo "where K is the keyword FST, L2 is a lexicon that contains pronunciations of"
  echo "keywords in K, E is an edit distance FST containing the confusion and L1 is the"
  echo "original lexicon."
  echo ""
  echo "Usage: local/generate_example_kws.sh <kws-data-dir> <L2-lexicon>"
  echo "                                     <L1-lexicon> <symbol-table>"
  echo " e.g.: local/generate_example_kws.sh data/kws oov_lexicon.txt"
  echo "                                     data/local/tmp.lang/lexiconp.txt"
  echo "                                     data/lang/words.txt"
  exit 1;
fi

# Parameters
kwsdatadir=$1; shift
keywords=$1; shift
L2_lexicon=$1; shift
L1_lexicon=$1; shift
symtab=$1; shift

mkdir -p $kwsdatadir/tmp

# You may modify the lexicon here; For example, we removed the stress marks for
# Tagalog lexicon. Besides, if the oov lexicon and original lexicon have the
# pronunciation for the same word, we use that in the original lexicon.
cat $L1_lexicon | sed 's/\s/ /g' | cut -d ' ' -f 1 |\
  tr '[:upper:]' '[:lower:]' |\
  paste - <(cat $L1_lexicon | sed 's/\s/ /g' | cut -d ' ' -f 2-) |\
  sed 's/_[B|E|I|S]//g' | sed 's/_[%|"]//g' |\
  awk '{if(NF>=2) {print $0}}' > $kwsdatadir/tmp/L1.lex
cat $L2_lexicon | sed 's/\s/ /g' | cut -d ' ' -f 1 |\
  tr '[:upper:]' '[:lower:]' |\
  paste - <(cat $L2_lexicon | sed 's/\s/ /g' | cut -d ' ' -f 2-) |\
  sed 's/_[B|E|I|S]//g' | sed 's/_[%|"]//g' |\
  awk '{if(NF>=2) {print $0}}' |\
  grep -v -f <(cat $kwsdatadir/tmp/L1.lex | awk '{print "^"$1"[\t| ]"}' | sort -u) |\
  cat - $kwsdatadir/tmp/L1.lex > $kwsdatadir/tmp/L2.lex


keywords_texts=$kwsdatadir/tmp/keywords.txt
# Removes short keywords that may lead to excessive false alarms.
cat $keywords | perl -e '
  open(W, "<'$kwsdatadir/tmp/L2.lex'") || die "Fail to open OOV lexicon: '$kwsdatadir/tmp/L2.lex'\n";
  my %lexicon;
  while (<W>) {
    chomp;
    my @col = split();
    @col >= 2 || die "Bad line in lexicon: $_\n";
    if ('$pron_probs' eq "false") {
      $lexicon{$col[0]} = scalar(@col)-1;
    } else {
      $lexicon{$col[0]} = scalar(@col)-2;
    }
  }
  while (<>) {
    chomp;
    my $line = $_;
    my @col = split();
    @col >= 2 || die "Bad line in keywords file: $_\n";
    my $len = 0;
    for (my $i = 1; $i < scalar(@col); $i ++) {
      if (defined($lexicon{$col[$i]})) {
        $len += $lexicon{$col[$i]};
      } else {
        print STEDRR "No pronunciation found for word: $col[$i]\n";
      }
    }
    if ($len >= '$phone_cutoff') {
      print "$line\n";
    } else {
      my $w=join(" ", @col[1..$#col]);
      print STDERR "Keyword $col[0] (text \"$w\") too short (cutoff='$phone_cutoff', len=$len)\n";
    }
  }' > $keywords_texts

# Gets phone symbols
phone_start=2
if [ $pron_probs ]; then
  phone_start=3
fi
cat $kwsdatadir/tmp/L2.lex $kwsdatadir/tmp/L1.lex |\
  awk '{for(i='$phone_start'; i <= NF; i++) {print $i;}}' | sort -u |\
  sed '1i\<eps>' | awk 'BEGIN{x=0} {print $0"\t"x; x++;}' > $kwsdatadir/tmp/phones.txt

# Gets word symbols; We append new words to the original word symbol table
max_id=`cat $symtab | awk '{print $2}' | sort -n | tail -1`;
cat $keywords_texts |\
  awk '{for(i=2; i <= NF; i++) {print $i;}}' |\
  cat - <(cat $kwsdatadir/tmp/L2.lex | awk '{print $1;}') |\
  cat - <(cat $kwsdatadir/tmp/L1.lex | awk '{print $1;}') | sort -u |\
  grep -F -v -x -f <(cat $symtab | awk '{print $1;}') |\
  awk 'BEGIN{x='$max_id'+1}{print $0"\t"x; x++;}' |\
  cat $symtab - | tr '[:upper:]' '[:lower:]' > $kwsdatadir/tmp/words.txt

# Compiles lexicon into FST
pron_probs_param="";
if [ $pron_probs ]; then
  pron_probs_param="--pron-probs";
fi
cat $kwsdatadir/tmp/L2.lex | utils/make_lexicon_fst.pl $pron_probs_param - |\
  fstcompile --isymbols=$kwsdatadir/tmp/phones.txt --osymbols=$kwsdatadir/tmp/words.txt - |\
  fstinvert | fstarcsort --sort_type=olabel > $kwsdatadir/tmp/L2.fst
cat $kwsdatadir/tmp/L1.lex | utils/make_lexicon_fst.pl $pron_probs_param - |\
  fstcompile --isymbols=$kwsdatadir/tmp/phones.txt --osymbols=$kwsdatadir/tmp/words.txt - |\
  fstarcsort --sort_type=ilabel > $kwsdatadir/tmp/L1.fst

# Compiles E.fst

confusion_matrix_param=""
if [ ! -z $confusion_matrix ]; then
  echo "$0: Using confusion matrix, normalizing"
  local/count_to_logprob.pl --cutoff $count_cutoff \
    $confusion_matrix $kwsdatadir/tmp/confusion.txt
  confusion_matrix_param="--confusion-matrix $kwsdatadir/tmp/confusion.txt"
fi
cat $kwsdatadir/tmp/phones.txt |\
  grep -v -E "<.*>" | grep -v "SIL" | awk '{print $1;}' |\
  local/build_edit_distance_fst.pl --boundary-off=true \
  $confusion_matrix_param - - |\
  fstcompile --isymbols=$kwsdatadir/tmp/phones.txt \
  --osymbols=$kwsdatadir/tmp/phones.txt - $kwsdatadir/tmp/E.fst

# Pre-composes L2 and E, for the sake of efficiency
fstcompose $kwsdatadir/tmp/L2.fst $kwsdatadir/tmp/E.fst |\
  fstarcsort --sort_type=olabel > $kwsdatadir/tmp/L2xE.fst

keywords=$kwsdatadir/keywords.int
# Prepares for parallelization
# maybe we should do full lexical expansion here(instead of sym2int)
# to get all possible combinations of in-lexicon words
cat $keywords_texts |\
  utils/sym2int.pl -f 2- $kwsdatadir/tmp/words.txt > $keywords

nof_keywords=`cat $keywords|wc -l`
if [ $nj -gt $nof_keywords ]; then
  nj=$nof_keywords
  echo "$0: Too many number of jobs, using $nj instead"
fi

# Generates the proxy keywords
mkdir -p $kwsdatadir/tmp/split/log
$cmd JOB=1:$nj $kwsdatadir/tmp/split/log/proxy.JOB.log \
  split -n l/JOB/$nj $keywords \| \
  generate-proxy-keywords --verbose=1 --cost-threshold=$beam --nBest=$nbest \
  $kwsdatadir/tmp/L2xE.fst $kwsdatadir/tmp/L1.fst ark:- ark:$kwsdatadir/tmp/split/proxy.JOB.fsts

proxy_fsts=""
for j in `seq 1 $nj`; do
  proxy_fsts="$proxy_fsts $kwsdatadir/tmp/split/proxy.$j.fsts"
done
cat $proxy_fsts > $kwsdatadir/keywords.fsts
