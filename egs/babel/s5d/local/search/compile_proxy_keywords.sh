#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
#               2012-2014  Guoguo Chen
# License: Apache 2.0

# Begin configuration section.
nj=8
cmd=run.pl
beam=-1             # Beam for proxy FST, -1 means no prune
phone_beam=-1       # Beam for KxL2xE FST, -1 means no prune
nbest=-1            # Use top n best proxy keywords in proxy FST, -1 means all
                    # proxies
phone_nbest=-1      # Use top n best phone sequences in KxL2xE, -1 means all
                    # phone sequences
confusion_matrix=   # If supplied, using corresponding E transducer
count_cutoff=1      # Minimal count to be considered in the confusion matrix;
                    # will ignore phone pairs that have count less than this.
pron_probs=true     # If true, then lexicon looks like:
                    # Word Prob Phone1 Phone2...
g_beam=10
g_alpha=
g_inv_alpha=
g2p_nbest=10
g2p_mass=0.95
case_insensitive=true
icu_transform="Any-Lower"
filter="OOV=1"

# End configuration section

echo "$0 " "$@"
. ./utils/parse_options.sh || exit 1;

# Gets phone symbols
phone_start=2
if $pron_probs; then
  phone_start=3
fi

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

data=$1
lang=$2
l1lex=$3
g2p=$4
workdir=$5

if [ ! -z "$g_inv_alpha" ] && [  $g_inv_alpha -ne 0 ] ; then
  g_alpha=$(echo print 1.0/$g_inv_alpha | perl )
fi

# Checks some files.
for f in $l1lex  $data/categories $data/keywords.txt ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1
done

mkdir -p $workdir
cat $data/categories | \
  local/search/filter_by_category.pl $data/categories "$filter" > $workdir/categories

grep -w -F -f <(awk '{print $1}' $workdir/categories) $data/keywords.txt |\
  sort -R > $workdir/keywords.filtered

paste <(cut -f 1  $workdir/keywords.filtered) \
      <(cut -f 2- $workdir/keywords.filtered | uconv -f utf-8 -t utf-8 -x "$icu_transform") >  $workdir/keywords.txt

cat $l1lex | perl -e '
  while (<>) {
    ($word, $prob, $pron) = split " ", $_, 3;
    $pron =~ s/_[^\s]+//g;
    $pron =~ s/\s+/ /g;
    $pron =~ s/^\s+//g;
    $pron =~ s/\s+$//g;
    print "$word $prob $pron\n"
  }
' | sort -u > $workdir/L1.lex

mkdir -p $workdir/lexicon

cat $workdir/keywords.txt | perl -e '
  open(f, shift @ARGV);
  while(<f>) {
    @F = split;
    $lex{$F[0]} = 1;
  }
  close(f);

  while(<STDIN>) {
    @F = split;
    foreach $w (@F[1..$#F]) {
      print "$w\n" unless defined $lex{$w};
    }
  }
' $workdir/L1.lex | sort -u > $workdir/lexicon/oov.txt

local/apply_g2p.sh --nj $nj --cmd "$cmd" --icu-transform "$icu_transform" \
    --var-counts $g2p_nbest --var-mass $g2p_mass \
    $workdir/lexicon/oov.txt $g2p $workdir/lexicon || exit 1

cat $workdir/L1.lex | \
  perl -e '
  while ( $line = <STDIN> ) {
    chomp $line;
    ($word, $pron) = split " ", $line, 2;
    $pron = join(" ", split(" ", $pron));
    push @{$LEX{$pron}}, $word;
  }

  open(L1, "| sort -u > $ARGV[0]") or die "Cannot open $ARGV[0]\n";
  open(MAP, "| sort -u > $ARGV[1]") or die "Cannot open $ARGV[1]\n";
  foreach $pron (keys %LEX) {
    $head = $LEX{$pron}->[0];
    print L1 "$head $pron\n";
    foreach $alt (@{$LEX{$pron}}) {
      print MAP "0 0 $alt $head\n";
    }
  }
  print MAP "0\n";
  close(L1);
  close(MAP);
' $workdir/L1.dedup.lex $workdir/L1.revdup.fst.txt

pron_probs_param=""
$pron_probs && pron_probs_param="--pron-probs"

# Creates words.txt that covers all the words in L1.lex and L2.lex. We append
# new words to the original word symbol table.
cat $workdir/L1.lex $workdir/lexicon/lexicon.lex | \
  perl -e '
    binmode STDIN, ":utf8";
    binmode STDOUT, ":utf8";
    binmode STDERR, ":utf8";
    $max_id=0;
    %WORDS=();
    open(F, "<:utf8" , $ARGV[0]) or die "Cannot open $ARGV[0]";
    while(<F>) {
      ($word, $id) = split(" ", $_);
      $WORDS{$word} = $id;
      $max_id = $id > $max_id ? $id : $max_id;
    }
    close(F);
    while (<STDIN>) {
      @F = split(" ", $_);
      if (not exists $WORDS{$F[0]}) {
        $WORDS{$F[0]} = $max_id + 1;
        $max_id += 1;
      }
    }
    foreach $kw (keys %WORDS) {
      print "$kw $WORDS{$kw}\n";
    }
  ' $lang/words.txt |  sort -k2,2n > $workdir/words.txt

cat $workdir/words.txt | \
  uconv -f utf-8 -t utf-8 -x "$icu_transform" >  $workdir/words.normalized.txt

#--ndisambig=`utils/add_lex_disambig.pl \
#--  $pron_probs_param $workdir/L1.dedup.lex $workdir/L1.disambig.lex`
#--ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
#--( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $workdir/disambig.txt

#remove all position dependent info and other tags
awk '{print $1;}' $lang/phones.txt | sed 's/_[BEIS]//g' | sed 's/_.*//g' | \
  grep -v '^#' | uniq |\
  perl -ne 'BEGIN{$i=0;}; chomp; print $_ . " " . $i . "\n"; $i+=1;' > $workdir/phones.txt

#--cat $workdir/L2.lex $workdir/L1.lex |\
#--  awk '{for(i='$phone_start'; i <= NF; i++) {print $i;}}' |\
#--  sort -u | sed '1i\<eps>' |\
#--  cat - $workdir/disambig.txt | awk 'BEGIN{x=0} {print $0"\t"x; x++;}' \
#--  > $workdir/phones.txt

cat $workdir/keywords.txt |\
  local/kwords2indices.pl --map-oov 0 $workdir/words.normalized.txt > $workdir/keywords.int


cat $workdir/L1.lex $workdir/lexicon/lexicon.lex | sed 's/\t/ /g' | \
  perl -ne 'chomp;
             ($word, $pron) = split / /, $_, 2;
             $pron =~ s/_[^ ]*//g;
             print "$word $pron\n";' | \
  sort -u > $workdir/L2.lex

cat $workdir/L1.revdup.fst.txt |\
  fstcompile --isymbols=$workdir/words.txt --osymbols=$workdir/words.txt - |\
  fstarcsort --sort_type=olabel - $workdir/L1.revdup.fst

echo ""

#--phone_disambig_symbol=`grep \#0 $workdir/phones.txt | awk '{print $2}'`
#--word_disambig_symbol=`grep \#0 $workdir/words.txt | awk '{print $2}'`
#--phone_disambig_symbols=`grep "^#" $workdir/phones.txt |\
#--  awk '{print $2}' | tr "\n" " "`
#--word_disambig_symbols=`grep "^#" $workdir/words.txt |\
#--  awk '{print $2}' | tr "\n" " "`
#--
#--cat $workdir/L1.disambig.lex |\
#--  utils/make_lexicon_fst.pl $pron_probs_param - |\
#--  fstcompile --isymbols=$workdir/phones.txt \
#--  --osymbols=$workdir/words.txt - |\
#--  fstaddselfloops "echo $phone_disambig_symbol |" \
#--  "echo $word_disambig_symbol |" |\
#--  fstdeterminize | fstrmsymbols "echo $phone_disambig_symbols|" |\
#--  fstrmsymbols --remove-from-output=true "echo $word_disambig_symbols|" |\
#--  fstarcsort --sort_type=ilabel > $workdir/L1.fst

cat $workdir/L1.dedup.lex |\
  utils/make_lexicon_fst.pl $pron_probs_param - |\
  fstcompile --isymbols=$workdir/phones.txt --osymbols=$workdir/words.txt - |\
  fstarcsort --sort_type=ilabel > $workdir/L1.fst

echo ""
cat $workdir/L2.lex  |\
  utils/make_lexicon_fst.pl $pron_probs_param - |\
  fstcompile --isymbols=$workdir/phones.txt --osymbols=$workdir/words.txt - |\
  fstinvert | fstarcsort --sort_type=olabel > $workdir/L2.fst

# Compiles E.fst
conf_mat_param=""
if [ ! -z $confusion_matrix ]; then
  echo "$0: Using confusion matrix, normalizing"
  local/count_to_logprob.pl --cutoff $count_cutoff \
    $confusion_matrix $workdir/confusion.txt
  conf_mat_param="--confusion-matrix $workdir/confusion.txt"
fi

cat $workdir/phones.txt | \
  grep -v -F -f $lang/phones/silence.txt | awk '{print $1;}' |\
  local/build_edit_distance_fst.pl --boundary-off=true $conf_mat_param - - |\
  fstcompile --isymbols=$workdir/phones.txt \
  --osymbols=$workdir/phones.txt - $workdir/E.fst

# Pre-composes L2 and E, for the sake of efficiency
$cmd --mem 12G $workdir/log/fstcompose.log \
  fstcompose $workdir/L2.fst $workdir/E.fst \|\
  fstarcsort --sort_type=ilabel \> $workdir/L2xE.fst

nof_keywords=`cat $workdir/keywords.txt |wc -l`
if [ $nj -gt $nof_keywords ]; then
  nj=$nof_keywords
  echo "$0: Too many number of jobs, using $nj instead"
fi

# Generates the proxy keywords
mkdir -p $workdir/split/log
if [ -z "$g_alpha" ] || [ $g_inv_alpha -eq 0 ] ; then
  echo "$0: Generating proxies without G.fst"
  $cmd JOB=1:$nj $workdir/split/log/proxy.JOB.log \
    split -n r/JOB/$nj $workdir/keywords.int \| \
    generate-proxy-keywords --verbose=1 \
    --proxy-beam=$beam --proxy-nbest=$nbest \
    --phone-beam=$phone_beam --phone-nbest=$phone_nbest \
    $workdir/L2xE.fst $workdir/L1.fst ark:- ark,t:$workdir/split/proxy.JOB.fsts
else
  echo "$0: Generating proxies with G.fst"
  $cmd JOB=1:$nj $workdir/split/log/proxy.JOB.log \
    split -n r/JOB/$nj $workdir/keywords.int \| \
    generate-proxy-keywords-ex --verbose=1 --g-beam=$g_beam --g-alpha=$g_alpha\
    --proxy-beam=$beam --proxy-nbest=$nbest \
    --phone-beam=$phone_beam --phone-nbest=$phone_nbest \
    $workdir/L2xE.fst $workdir/L1.fst $lang/G.fst ark:- ark,t:$workdir/split/proxy.JOB.fsts
fi


proxy_fsts=""
for j in `seq 1 $nj`; do
  proxy_fsts="$proxy_fsts $workdir/split/proxy.$j.fsts"
done
cat $proxy_fsts | fsttablecompose $workdir/L1.revdup.fst ark:- ark:- |\
  fsts-project ark:- ark,scp:$workdir/keywords.fsts,-|\
  sort -o $workdir/keywords.scp
