#!/usr/bin/env bash

# Copyright 2017 Xingyu Na
# Apache 2.0

# prepare dict resources

. ./path.sh

[ $# != 1 ] && echo "Usage: $0 <resource-path>" && exit 1;

res_dir=$1
dict_dir=data/local/dict
mkdir -p $dict_dir
cp $res_dir/lexicon.txt $dict_dir

cat $dict_dir/lexicon.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
  perl -e 'while(<>){ chomp($_); $phone = $_; next if ($phone eq "sil");
    m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$1} .= "$phone "; }
    foreach $l (values %q) {print "$l\n";}
  ' | sort -k1 > $dict_dir/nonsilence_phones.txt  || exit 1;

echo sil > $dict_dir/silence_phones.txt

echo sil > $dict_dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone

cat $dict_dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dict_dir/extra_questions.txt || exit 1;
cat $dict_dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dict_dir/extra_questions.txt || exit 1;

echo "$0: AISHELL dict preparation succeeded"
exit 0;
