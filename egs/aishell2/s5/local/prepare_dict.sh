#!/usr/bin/env bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# This is a shell script, and it download and process DaCiDian for Mandarin ASR.

. ./path.sh

download_dir=data/local/DaCiDian
dir=data/local/dict

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dict-dir>";
  exit 1;
fi

dir=$1

# download the DaCiDian from github
if [ ! -d $download_dir ]; then
  git clone https://github.com/aishell-foundation/DaCiDian.git $download_dir
fi

# here we map <UNK> to the phone spn(spoken noise)
mkdir -p $dir
python $download_dir/DaCiDian.py $download_dir/word_to_pinyin.txt $download_dir/pinyin_to_phone.txt > $dir/lexicon.txt
echo -e "<UNK>\tspn" >> $dir/lexicon.txt

# prepare silence_phones.txt, nonsilence_phones.txt, optional_silence.txt, extra_questions.txt
cat $dir/lexicon.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
  perl -e 'while(<>){ chomp($_); $phone = $_; next if ($phone eq "sil");
    m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$1} .= "$phone "; }
    foreach $l (values %q) {print "$l\n";}
  ' | sort -k1 > $dir/nonsilence_phones.txt  || exit 1;

echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

cat $dir/silence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; if($p eq "\$0"){$q{""} .= "$p ";}else{$q{$2} .= "$p ";} } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

echo "local/prepare_dict.sh succeeded"
exit 0;
