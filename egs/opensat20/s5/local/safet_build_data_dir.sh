#!/bin/bash
# Copyright (c) 2020, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

INPUT=$1
TRANSCRIPT=$2
OUTPUT=$1

cat $TRANSCRIPT | awk '{print $1, $2, $3/100, $4/100}' | sort > $OUTPUT/segments
cat $TRANSCRIPT | awk '{printf $1""FS;for(i=6; i<=NF; ++i) printf "%s",$i""FS; print""}' | sort > $OUTPUT/text
cat $TRANSCRIPT | awk '{print $1, $5}' | sort  > $OUTPUT/utt2spk
utils/utt2spk_to_spk2utt.pl < $OUTPUT/utt2spk >$OUTPUT/spk2utt


utils/fix_data_dir.sh $OUTPUT
utils/validate_data_dir.sh --no-feats $OUTPUT


