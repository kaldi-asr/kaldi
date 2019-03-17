#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2018  Yuan-Fu Liao, National Taipei University of Technology
# Apache 2.0

source_dir=NER-Trs-Vol1/Language
dict_dir=data/local/dict
rm -rf $dict_dir
mkdir -p $dict_dir

#
#
#
rm -f $dict_dir/lexicon.txt
touch $dict_dir/lexicon.txt
cat $source_dir/lexicon.txt > $dict_dir/lexicon.txt
echo "<SIL> SIL"	>> $dict_dir/lexicon.txt

#
# define silence phone
#
rm -f $dict_dir/silence_phones.txt
touch $dict_dir/silence_phones.txt

echo "SIL"	> $dict_dir/silence_phones.txt

#
# find nonsilence phones
#
rm -f $dict_dir/nonsilence_phones.txt
touch $dict_dir/nonsilence_phones.txt

cat $source_dir/lexicon.txt | grep -v -F -f $dict_dir/silence_phones.txt | \
    perl -ane 'print join("\n", @F[1..$#F]) . "\n"; '  | \
    sort -u > $dict_dir/nonsilence_phones.txt

#
# add optional silence phones
#

rm -f $dict_dir/optional_silence.txt
touch $dict_dir/optional_silence.txt
echo "SIL"	> $dict_dir/optional_silence.txt

#
# extra questions
#
rm -f $dict_dir/extra_questions.txt
touch $dict_dir/extra_questions.txt
cat $dict_dir/silence_phones.txt    | awk '{printf("%s ", $1);} END{printf "\n";}'  > $dict_dir/extra_questions.txt || exit 1;
cat $dict_dir/nonsilence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' >> $dict_dir/extra_questions.txt || exit 1;

echo "Dictionary preparation succeeded"
exit 0;
