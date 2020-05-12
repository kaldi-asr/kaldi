#!/bin/bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# This is a shell script, and it download and process DaCiDian for Mandarin ASR.

. ./path.sh

dir=data/local/dict

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dict-dir>";
  exit 1;
fi

dir=$1


# prepare silence_phones.txt, nonsilence_phones.txt, optional_silence.txt, extra_questions.txt
awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' $dir/lexicon.txt | \
  perl -e 'while(<>){ chomp($_); $phone = $_; next if ($phone eq "sil");
    m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$1} .= "$phone "; }
    foreach $l (values %q) {print "$l\n";}
  ' | sort -k1 > $dir/nonsilence_phones.txt  || exit 1;

echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

awk '{printf("%s ", $1);} END{printf "\n";}' $dir/silence_phones.txt > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; if($p eq "\$0"){$q{""} .= "$p ";}else{$q{$2} .= "$p ";} } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

echo "local/prepare_dict.sh succeeded"
exit 0;
