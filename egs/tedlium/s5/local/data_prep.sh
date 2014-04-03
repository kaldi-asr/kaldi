#!/bin/bash
#

# To be run from one directory above this script.

. path.sh

export LC_ALL=C


dir=data/local/train

mkdir -p $dir

cat db/train/stm/*.stm | sed "s:\([A-Z]\) ':\1':g" > $dir/stm.txt

cat $dir/stm.txt | sed 's:<sil>::g' | sed 's:([0-9])::g' |
  awk '{printf ("%s ", $NF); for (i=7;i<NF;i++) printf("%s ", $i); printf("\n");}' |
  tr '{}' '[]' | tr -d '()' | sort | local/join.py db/TEDLIUM.150K.dic > $dir/text

cat $dir/text | cut -d" " -f 1 | awk -F"-" '{print $0, $1, $2, $3}' > $dir/segments

cat $dir/segments | awk '{print $1, $2}' > $dir/utt2spk

cat $dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt

cat $dir/spk2utt | cut -d" " -f 1 |
    awk '{print $0, "sph2pipe -f wav -p db/train/sph/" $0 ".sph |" }' > $dir/wav.scp

dir=data/local/test

mkdir -p $dir

cat db/test/stm/*.stm | grep -v ignore_time_segment_in_scoring |
 sed "s:\([A-Z]\) ':\1':g" > $dir/stm.txt

cat $dir/stm.txt | sed 's:<sil>::g' | sed 's:([0-9])::g' |
  awk '{printf ("%s-%s-%s ", $1, $4, $5); for (i=7;i<=NF;i++) printf("%s ", $i); printf("\n");}' |
  tr '{}' '[]' | tr -d '()' | sort | local/join.py db/TEDLIUM.150K.dic > $dir/text

cat $dir/text | cut -d" " -f 1 | awk -F"-" '{print $0, $1, $2, $3}' > $dir/segments

cat $dir/segments | awk '{print $1, $2}' > $dir/utt2spk

cat $dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt

cat $dir/spk2utt | cut -d" " -f 1 |
    awk '{print $0, "sph2pipe -f wav -p db/test/sph/" $0 ".sph |" }' > $dir/wav.scp

