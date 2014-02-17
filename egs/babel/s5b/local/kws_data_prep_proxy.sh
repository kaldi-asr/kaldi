#!/bin/bash

# Copyright 2014  Guoguo Chen
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
phone_cutoff=5      # We don't generate proxy keywords for OOV keywords that
                    # have less phones than the specified cutoff as they may
                    # introduce a lot false alarms
confusion_matrix=   # If supplied, using corresponding E transducer
count_cutoff=1      # Minimal count to be considered in the confusion matrix;
                    # will ignore phone pairs that have count less than this.
pron_probs=false    # If true, then lexicon looks like:
                    # Word Prob Phone1 Phone2...
case_insensitive=true
icu_transform="Any-Lower"
proxy_set=          # List of keywords to generate proxies for, one KWID per
                    # line. If empty, then by default generate proxies for all
                    # OOV keywords.
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
echo $0 "$@"
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: local/kws_data_prep_proxy.sh <lang-dir> <data-dir> \\"
  echo "                 <L1-lexicon> <L2-lexicon> <kws-data-dir>"
  echo " e.g.: local/kws_data_prep_proxy.sh data/lang/ data/dev10h/ \\"
  echo "      data/local/tmp.lang/lexiconp.txt oov_lexicon.txt data/dev10h/kws/"
  echo "allowed options:"
  echo "  --case-sensitive <true|false>  # Being case-sensitive or not"
  echo "  --icu-transform  <string>      # Transliteration for upper/lower case" 
  echo "                                 # mapping"
  echo "  --proxy-set      <IV/OOV>      # Keyword set for generating proxies"
  exit 1
fi

set -e 
set -o pipefail

langdir=$1
datadir=$2
l1_lexicon=$3
l2_lexicon=$4
kwsdatadir=$5

# Checks some files.
for f in $langdir/words.txt $kwsdatadir/kwlist.xml $l1_lexicon $l2_lexicon; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1
done

keywords=$kwsdatadir/kwlist.xml
mkdir -p $kwsdatadir/tmp/

cat $keywords | perl -e '
  #binmode STDIN, ":utf8"; 
  binmode STDOUT, ":utf8"; 

  use XML::Simple;
  use Data::Dumper;

  my $data = XMLin(\*STDIN);

  #print Dumper($data->{kw});
  foreach $kwentry (@{$data->{kw}}) {
    #print Dumper($kwentry);
    print "$kwentry->{kwid}\t$kwentry->{kwtext}\n";
  }' > $kwsdatadir/raw_keywords_all.txt

# Takes care of upper/lower case.
cp $langdir/words.txt $kwsdatadir/words.txt
cat $l1_lexicon | sed 's/\s/ /g' > $kwsdatadir/tmp/L1.tmp.lex
if $case_insensitive; then
  echo "$0: Running case insensitive processing"
  echo "$0: Using ICU with transofrm \"$icu_transform\""

  # Processing words.txt
  cat $kwsdatadir/words.txt |\
    uconv -f utf8 -t utf8 -x "${icu_transform}"  > $kwsdatadir/words.norm.txt

  # Processing lexicon
  cat $l2_lexicon | sed 's/\s/ /g' | cut -d ' ' -f 1 |\
    uconv -f utf8 -t utf8 -x "${icu_transform}" |\
    paste -d ' ' - <(cat $l2_lexicon | sed 's/\s/ /g' | cut -d ' ' -f 2-) \
    > $kwsdatadir/tmp/L2.tmp.lex

  paste <(cut -f 1 $kwsdatadir/raw_keywords_all.txt) \
    <(cut -f 2 $kwsdatadir/raw_keywords_all.txt |\
    uconv -f utf8 -t utf8 -x "${icu_transform}") \
    > $kwsdatadir/keywords_all.txt
  cat $kwsdatadir/keywords_all.txt |\
    local/kwords2indices.pl --map-oov 0 $kwsdatadir/words.norm.txt \
    > $kwsdatadir/keywords_all.int
else
  cat $l2_lexicon | sed 's/\s/ /g' > $kwsdatadir/tmp/L2.tmp.lex
  cp $kwsdatadir/raw_keywords_all.txt $kwsdatadir/keywords_all.txt
  
  cat $kwsdatadir/keywords_all.txt | \
    sym2int.pl --map-oov 0 -f 2- $kwsdatadir/words.txt \
    > $kwsdatadir/keywords_all.int
fi

# Writes some scoring related files.
cat $kwsdatadir/keywords_all.int |\
  egrep -v " 0 | 0$" | cut -f 1 -d ' ' |\
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_invocab.xml

cat $kwsdatadir/keywords_all.int |\
  egrep " 0 | 0$" | cut -f 1 -d ' ' |\
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_outvocab.xml

# Selects a set to generate proxies for. By default, generate proxies for OOV
# keywords.
if [ -z $proxy_set ]; then
  cat $kwsdatadir/keywords_all.int |\
    egrep " 0 | 0$" | awk '{print $1;}' | sort -u \
    > $kwsdatadir/keywords_proxy.list
else
  cp $proxy_set $kwsdatadir/keywords_proxy.list
fi
cat $kwsdatadir/keywords_all.txt |\
  grep -f $kwsdatadir/keywords_proxy.list > $kwsdatadir/keywords_proxy.txt
cat $kwsdatadir/keywords_proxy.txt |\
  cut -f 2- | awk '{for(x=1;x<=NF;x++) {print $x;}}' |\
  sort -u > $kwsdatadir/keywords_proxy_words.list

# Maps original phone set to a "reduced" phone set. We limit L2 to only cover
# the words that are actually used in keywords_proxy.txt for efficiency purpose.
# Besides, if L1 and L2 contains the same words, we use the pronunciation from
# L1 since it is the lexicon used for the LVCSR training.
cat $kwsdatadir/tmp/L1.tmp.lex | cut -d ' ' -f 1 |\
  paste -d ' ' - <(cat $kwsdatadir/tmp/L1.tmp.lex | cut -d ' ' -f 2-|\
  sed 's/_[B|E|I|S]//g' | sed 's/_[%|"]//g') |\
  awk '{if(NF>=2) {print $0}}' > $kwsdatadir/tmp/L1.lex
cat $kwsdatadir/tmp/L2.tmp.lex | cut -d ' ' -f 1 |\
  paste -d ' ' - <(cat $kwsdatadir/tmp/L2.tmp.lex | cut -d ' ' -f 2-|\
  sed 's/_[B|E|I|S]//g' | sed 's/_[%|"]//g') |\
  awk '{if(NF>=2) {print $0}}' | perl -e '
  ($lex1, $words) = @ARGV;
  open(L, "<$lex1") || die "Fail to open $lex1.\n";
  open(W, "<$words") || die "Fail to open $words.\n";
  while (<L>) {
    chomp;
    @col = split;
    @col >= 2 || die "Too few columsn in \"$_\".\n";
    $w = $col[0];
    $w_p = $_;
    if (defined($lex1{$w})) {
      push(@{$lex1{$w}}, $w_p);
    } else {
      $lex1{$w} = [$w_p];
    }
  }
  close(L);
  while (<STDIN>) {
    chomp;
    @col = split;
    @col >= 2 || die "Too few columsn in \"$_\".\n";
    $w = $col[0];
    $w_p = $_;
    if (defined($lex1{$w})) {
      next;
    }
    if (defined($lex2{$w})) {
      push(@{$lex2{$w}}, $w_p);
    } else {
      $lex2{$w} = [$w_p];
    }
  }
  %lex = (%lex1, %lex2);
  while (<W>) {
    chomp;
    if (defined($lex{$_})) {
      foreach $x (@{$lex{$_}}) {
        print "$x\n";
      }
    }
  }
  close(W);
  ' $kwsdatadir/tmp/L1.lex $kwsdatadir/keywords_proxy_words.list \
  > $kwsdatadir/tmp/L2.lex
rm -f $kwsdatadir/tmp/L1.tmp.lex $kwsdatadir/tmp/L2.tmp.lex

# Creates words.txt that covers all the words in L1.lex and L2.lex. We append
# new words to the original word symbol table.
max_id=`cat $kwsdatadir/words.txt | awk '{print $2}' | sort -n | tail -1`;
cat $kwsdatadir/keywords_proxy.txt |\
  awk '{for(i=2; i <= NF; i++) {print $i;}}' |\
  cat - <(cat $kwsdatadir/tmp/L2.lex | awk '{print $1;}') |\
  cat - <(cat $kwsdatadir/tmp/L1.lex | awk '{print $1;}') |\
  sort -u | grep -F -v -x -f <(cat $kwsdatadir/words.txt | awk '{print $1;}') |\
  awk 'BEGIN{x='$max_id'+1}{print $0"\t"x; x++;}' |\
  cat $kwsdatadir/words.txt - > $kwsdatadir/tmp/words.txt

# Creates keyword list that we need to generate proxies for.
cat $kwsdatadir/keywords_proxy.txt | perl -e '
  open(W, "<'$kwsdatadir/tmp/L2.lex'") ||
    die "Fail to open L2 lexicon: '$kwsdatadir/tmp/L2.lex'\n";
  my %lexicon;
  while (<W>) {
    chomp;
    my @col = split();
    @col >= 2 || die "'$0': Bad line in lexicon: $_\n";
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
        print STEDRR "'$0': No pronunciation found for word: $col[$i]\n";
      }
    }
    if ($len >= '$phone_cutoff') {
      print "$line\n";
    } else {
      print STDERR "'$0': Keyword $col[0] is too short, not generating proxy\n";
    }
  }' > $kwsdatadir/tmp/keywords.txt

# Creates proxy keywords.
local/generate_proxy_keywords.sh \
  --cmd "$cmd" --nj "$nj" --beam "$beam" --nbest "$nbest" \
  --phone-beam $phone_beam --phone-nbest $phone_nbest \
  --confusion-matrix "$confusion_matrix" --count-cutoff "$count_cutoff" \
  --pron-probs "$pron_probs" $kwsdatadir/tmp/
cp $kwsdatadir/tmp/keywords.fsts $kwsdatadir

# Creates utterance id for each utterance.
cat $datadir/segments | \
  awk '{print $1}' | \
  sort | uniq | perl -e '
  $idx=1;
  while(<>) {
    chomp;
    print "$_ $idx\n";
    $idx++;
  }' > $kwsdatadir/utter_id

# Map utterance to the names that will appear in the rttm file. You have 
# to modify the commands below accoring to your rttm file
cat $datadir/segments | awk '{print $1" "$2}' |\
  sort | uniq > $kwsdatadir/utter_map;

echo "$0: Kws data preparation succeeded"
