#!/usr/bin/env bash
# Copyright 2018 ASLP, NWPU (Author: Hang Lyu)
#
# Apache 2.0

# This is a shell script, and it download and process BigCiDian


download_dir=data/local/BigCiDian
dir=


. ./path.sh
. ./utils/parse_options.sh


if [ $# -ne 1 ]; then
  echo "Usage: $0 <dict-dir>"
  echo "e.g.: $0 data/local/dict"
  echo "This script downloads the BigCiDian and prepares the related files."
  exit 1;
fi

dir=$1

# download the BigCiDian from github
if [ ! -d $download_dir ]; then
  git clone https://github.com/speechio/BigCiDian.git $download_dir
fi

# compile the BigCiDian
cd $download_dir
(
  export LC_ALL="zh_CN.UTF-8"
  sh run.sh
)
cd -

# here we map <UNK> to the phone spn(spoken noise)
mkdir -p $dir
cp $download_dir/lexicon.txt $dir/lexicon_raw.txt
cp $download_dir/phoneset.list $dir/nonsilence_phones.txt

# add the silence phone part
echo -e "<UNK>\tspn" >> $dir/lexicon_raw.txt
echo -e "!SIL\tsil" >> $dir/lexicon_raw.txt  # place this item in case generate
                                             # silence word information in the
                                             # future
cat $dir/lexicon_raw.txt | sort | uniq > $dir/lexicon.txt

# prepare the silence_phones and optional_silence.txt
echo sil > $dir/silence_phones.txt
echo spn >> $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# prepare the extra_questions
cat $dir/silence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e '
  while(<>){ foreach $p (split(" ", $_)) {
    $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_";
    if ($p eq "\$0") {$q{""} .= "$p ";}
    else{ $q{$2} .= "$p ";} } }
  foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

# prepare a word_seg_lexicon.txt for word segmentation
# the entry's format of jieba is "word frequency", set the "frequency" to 99
awk '{print $1,99}' $dir/lexicon.txt > $dir/word_seg_lexicon.txt

echo "$0: Successed."
exit 0;
