#!/bin/bash
#

# To be run from one directory above this script.

. path.sh

export LC_ALL=C

for set in test train; do

dir=data/$set

mkdir -p $dir

cat db/TEDLIUM_release1/$set/stm/*.stm | grep -v ignore_time_segment_in_scoring |
   sed "s:\([A-Z]\) ':\1':g" > $dir/stm.txt

# Test set is a bit different
if [ "$set" = "train" ] ; then
cat $dir/stm.txt | sed 's:<sil>::g' | sed 's:([0-9])::g' |
  awk '{printf ("%s ", $NF); for (i=7;i<NF;i++) printf("%s ", $i); printf("\n");}' |
  tr '{}' '[]' | tr -d '()' | sort | local/join_suffix.py db/TEDLIUM_release1/TEDLIUM.150K.dic > $dir/text
else
cat $dir/stm.txt | sed 's:<sil>::g' | sed 's:([0-9])::g' |
  awk '{printf ("%s-%s-%s ", $1, $4, $5); for (i=7;i<=NF;i++) printf("%s ", $i); printf("\n");}' |
  tr '{}' '[]' | tr -d '()' | sort | local/join_suffix.py db/TEDLIUM_release1/TEDLIUM.150K.dic > $dir/text
fi


cat $dir/text | cut -d" " -f 1 | awk -F"-" '{print $0, $1, $2, $3}' > $dir/segments

cat $dir/segments | awk '{print $1, $2}' > $dir/utt2spk

cat $dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt

cat $dir/spk2utt | cut -d" " -f 1 |
    awk -v set="$set" '{print $0, "sph2pipe -f wav -p db/TEDLIUM_release1/" set "/sph/" $0 ".sph |" }' > $dir/wav.scp

done

