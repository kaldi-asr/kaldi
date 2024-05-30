#!/usr/bin/env bash
#
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Takes no arguments. 


tmpdir=data/local/tmp
[ ! -f $tmpdir/G.txt ] && echo "No such file $tmpdir/G.txt" && exit 1;

. ./path.sh || exit 1; # for KALDI_ROOT

mkdir -p data/lang_ug
cp -r data/lang/* data/lang_ug
rm -rf data/lang_ug/tmp

cat data/train/text  | \
    perl -e 'while(<>) { @A = split; shift @A; foreach $w(@A) { $tot_count++; $count{$w}++; } $n_sent++; } 
    $tot_count += $n_sent;
    foreach $k (keys %count) { $p = $count{$k} / $tot_count; $cost = -log($p); print "0  0  $k  $k  $cost\n"; }
    $final_cost = -log($n_sent / $tot_count);
    print "0 $final_cost\n"; ' | \
  fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false | fstarcsort --sort_type=ilabel > data/lang_ug/G.fst || exit 1;

# Checking that G is stochastic [note, it wouldn't be for an Arpa]
fstisstochastic data/lang_ug/G.fst || echo Error: G is not stochastic

# Checking that G.fst is determinizable.
fstdeterminize data/lang_ug/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_ug/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
fsttablecompose data/lang_ug/L_disambig.fst data/lang_ug/G.fst | \
   fstdeterminize >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang_ug/L.fst data/lang_ug/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.

# Checking that L_disambig.G is stochastic:
fsttablecompose data/lang_ug/L_disambig.fst data/lang_ug/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.

echo "Succeeded preparing grammar for RM."
