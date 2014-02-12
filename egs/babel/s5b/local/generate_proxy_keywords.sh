#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

nj=32
cmd=run.pl
beam=5                    # Beam for proxy FST; usually used together with the nbest option
nbest=100                 # First n best proxy keywords
phone_cutoff=5            # We don't generate proxy keywords for OOV keywords that have less phones
                          # than the specified cutoff; they may introduce more false alarms
count_cutoff=1            # Cutoff for the phone confusion pair counts 
confusion_matrix=

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Generate proxy keywords for OOV keywords. You may apply the confusion matrix. If you"
  echo "are going to use the confusion matrix, please use the following format for the file"
  echo "\$confusion_matrix:"
  echo "          p1 p2 count1        // For substitution"
  echo "          p3 <eps> count2     // For deletion"
  echo "          <eps> p4 count3     // For insertion"
  echo ""
  echo "Usage: local/generate_example_kws.sh <kws-data-dir> <oov-lexicon>"
  echo "                                     <lexicon> <symbol-table>"
  echo " e.g.: local/generate_example_kws.sh data/kws oov_lexicon.txt"
  echo "                                     data/local/lexicon.txt data/lang/words.txt"
  exit 1;
fi

# Parameters
kwsdatadir=$1
oov_lexicon=$2
original_lexicon=$3
original_symtab=$4

mkdir -p $kwsdatadir/tmp

# You may modify the lexicon here; For example, I removed the stress marks for the 
# Tagalog lexicon
cat $oov_lexicon |\
  sed 's/_[%|"]//g' | awk '{if(NF>=2) {print $0}}' > $kwsdatadir/tmp/oov.lex
cat $original_lexicon |\
  sed 's/_[%|"]//g' | awk '{if(NF>=2) {print $0}}' > $kwsdatadir/tmp/original.lex

# Get OOV keywords, and remove the short OOV keywords. Generate proxy keywords based
# on the phone confusion for the short OOV keywords may introduce a lot of false alarms,
# therefore we provide the cutoff option.
cat $kwsdatadir/kwlist_outvocab.xml | \
  grep -o -P "(?<=kwid=\").*(?=\")" |\
  paste - <(cat $kwsdatadir/kwlist_outvocab.xml | grep -o -P "(?<=<kwtext>).*(?=</kwtext>)") \
  > $kwsdatadir/tmp/oov_all.txt
cat $kwsdatadir/tmp/oov_all.txt | perl -e '
  open(W, "<'$kwsdatadir/tmp/oov.lex'") || die "Fail to open OOV lexicon: '$kwsdatadir/tmp/oov.lex'\n";
  my %lexicon;
  while (<W>) {
    chomp;
    my @col = split();
    @col >= 2 || die "Bad line in lexicon: $_\n";
    $lexicon{$col[0]} = scalar(@col)-1;
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
    }
  }' > $kwsdatadir/tmp/oov.txt

# Get phone symbols
cat $kwsdatadir/tmp/oov.lex $kwsdatadir/tmp/original.lex |\
  awk '{for(i=2; i <= NF; i++) {print $i;}}' | sort -u |\
  sed '1i\<eps>' | awk 'BEGIN{x=0} {print $0"\t"x; x++;}' > $kwsdatadir/tmp/phones.txt

# Get word symbols; We append new words to the original word symbol table
max_id=`cat $original_symtab | awk '{print $2}' | sort -n | tail -1`;
cat $kwsdatadir/tmp/oov.txt |\
  awk '{for(i=2; i <= NF; i++) {print $i;}}' |\
  cat - <(cat $kwsdatadir/tmp/oov.lex | awk '{print $1;}') |\
  cat - <(cat $kwsdatadir/tmp/original.lex | awk '{print $1}') | sort -u |\
  grep -F -v -x -f <(cat $original_symtab | awk '{print $1;}') |\
  awk 'BEGIN{x='$max_id'+1}{print $0"\t"x; x++;}' |\
  cat $original_symtab - > $kwsdatadir/tmp/words.txt

# Compile lexicon into FST
cat $kwsdatadir/tmp/oov.lex | utils/make_lexicon_fst.pl - |\
  fstcompile --isymbols=$kwsdatadir/tmp/phones.txt --osymbols=$kwsdatadir/tmp/words.txt - |\
  fstinvert | fstarcsort --sort_type=olabel > $kwsdatadir/tmp/oov_lexicon.fst
cat $kwsdatadir/tmp/original.lex | utils/make_lexicon_fst.pl - |\
  fstcompile --isymbols=$kwsdatadir/tmp/phones.txt --osymbols=$kwsdatadir/tmp/words.txt - |\
  fstarcsort --sort_type=ilabel > $kwsdatadir/tmp/original_lexicon.fst

# Compile E.fst
if [ -z $confusion_matrix ]; then
  cat $kwsdatadir/tmp/phones.txt |\
    grep -v -E "<.*>" | grep -v "SIL" | awk '{print $1;}' |\
    local/build_edit_distance_fst.pl --boundary-off=false - - |\
    fstcompile --isymbols=$kwsdatadir/tmp/phones.txt --osymbols=$kwsdatadir/tmp/phones.txt - $kwsdatadir/tmp/Edit.fst
else
  echo "$0: Using confusion matrix."
  local/count_to_logprob.pl --cutoff $count_cutoff $confusion_matrix $kwsdatadir/tmp/confusion.txt
  cat $kwsdatadir/tmp/phones.txt |\
    grep -v -E "<.*>" | grep -v "SIL" | awk '{print $1;}' |\
    local/build_edit_distance_fst.pl --boundary-off=false \
    --confusion-matrix=$kwsdatadir/tmp/confusion.txt - - |\
    fstcompile --isymbols=$kwsdatadir/tmp/phones.txt --osymbols=$kwsdatadir/tmp/phones.txt - $kwsdatadir/tmp/Edit.fst
fi

# Pre-compose L2 and E, for the sake of efficiency
fstcompose $kwsdatadir/tmp/oov_lexicon.fst $kwsdatadir/tmp/Edit.fst |\
  fstarcsort --sort_type=olabel > $kwsdatadir/tmp/L2xE.fst

# Prepare for parallelization
mkdir -p $kwsdatadir/tmp/split/
cat $kwsdatadir/tmp/oov.txt | utils/sym2int.pl -f 2- $kwsdatadir/tmp/words.txt > $kwsdatadir/tmp/oov.int
if [ $nj -gt `cat $kwsdatadir/tmp/oov.int | wc -l` ]; then
  nj=`cat $kwsdatadir/tmp/oov.int | wc -l`
  echo "$0: Too many number of jobs, using $nj instead"
fi
for j in `seq 1 $nj`; do
  let "id=$j-1";
  utils/split_scp.pl -j $nj $id $kwsdatadir/tmp/oov.int $kwsdatadir/tmp/split/$j.int
done

# Generate the proxy keywords
$cmd JOB=1:$nj $kwsdatadir/tmp/split/JOB.log \
  generate-proxy-keywords --verbose=1 \
  --cost-threshold=$beam --nBest=$nbest \
  $kwsdatadir/tmp/L2xE.fst $kwsdatadir/tmp/original_lexicon.fst \
  ark:$kwsdatadir/tmp/split/JOB.int ark:$kwsdatadir/tmp/split/JOB.fsts

# Post process
if [ ! -f $kwsdatadir/keywords_invocab.fsts ]; then
  cp -f $kwsdatadir/keywords.fsts $kwsdatadir/keywords_invocab.fsts
fi
cat $kwsdatadir/tmp/split/*.fsts > $kwsdatadir/keywords_outvocab.fsts
cat $kwsdatadir/keywords_invocab.fsts $kwsdatadir/keywords_outvocab.fsts \
  > $kwsdatadir/keywords.fsts
