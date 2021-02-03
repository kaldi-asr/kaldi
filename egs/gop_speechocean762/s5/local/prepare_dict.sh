#!/usr/bin/env bash

# Copyright      2014  Vassil Panayotov
#           2020-2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

# Prepares input directory of `utils/prepare_lang.sh`


. utils/parse_options.sh || exit 1;
. ./path.sh || exit 1


if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <lexicion> <dst-dir>"
  echo "e.g.: data/local/lexicon.txt data/local/dict_nosp"
  exit 1
fi

lexicon=$1
dst_dir=$2

lexicon_raw_nosil=$dst_dir/lexicon_raw_nosil.txt

mkdir -p $dst_dir || exit 1;

cp $lexicon $lexicon_raw_nosil || exit 1

silence_phones=$dst_dir/silence_phones.txt
optional_silence=$dst_dir/optional_silence.txt
nonsil_phones=$dst_dir/nonsilence_phones.txt
extra_questions=$dst_dir/extra_questions.txt

echo "Preparing phone lists"
(echo SIL; echo SPN;) > $silence_phones
echo SIL > $optional_silence
# nonsilence phones; on each line is a list of phones that correspond
# really to the same base phone.
awk '{for (i=2; i<=NF; ++i) { print $i; gsub(/[0-9]/, "", $i); print $i}}' $lexicon_raw_nosil |\
  sort -u |\
  perl -e 'while(<>){
    chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_";
    $phones_of{$1} .= "$_ "; }
    foreach $list (values %phones_of) {print $list . "\n"; } ' | sort \
    > $nonsil_phones || exit 1;
# A few extra questions that will be added to those obtained by automatically clustering
# the "real" phones. These ask about stress; there's also one for silence.
cat $silence_phones| awk '{printf("%s ", $1);} END{printf "\n";}' > $extra_questions || exit 1;
cat $nonsil_phones | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
  >> $extra_questions || exit 1;
echo "$(wc -l <$silence_phones) silence phones saved to: $silence_phones"
echo "$(wc -l <$optional_silence) optional silence saved to: $optional_silence"
echo "$(wc -l <$nonsil_phones) non-silence phones saved to: $nonsil_phones"
echo "$(wc -l <$extra_questions) extra triphone clustering-related questions saved to: $extra_questions"

(echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |\
cat - $lexicon_raw_nosil | sort | uniq >$dst_dir/lexicon.txt
echo "Lexicon text file saved as: $dst_dir/lexicon.txt"

exit 0
