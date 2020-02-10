#!/usr/bin/env bash

# Copyright 2019 (author: Jinyi Yang)
# Apache 2.0

# This scripts remove bad utterances from the tdt corpus.
. path.sh
. ./utils/parse_options.sh
if [ $# != 2 ]; then
   echo "Usage: $0 [options] <tmp_data_dir> <tgt_data_dir>";
   echo "e.g.: $0 TDT2 data/local/tdt2"
   exit 1;
fi

set -e -o pipefail

tdtdir=$1
tgtdir=$2
mkdir -p $tgtdir


for f in "text" "utt2spk" "segments" "uttid"; do
	cat $tdtdir/txt/$f | grep -v -F -f local/tdt_mandarin_bad_utts > $tgtdir/$f
done

awk 'NR==FNR{a[$2];next} $1 in a{print $0}' $tgtdir/segments $tdtdir/wav.scp | \
grep -v -F -f local/tdt_mandarin_bad_utts > $tgtdir/wav.scp

utils/utt2spk_to_spk2utt.pl $tgtdir/utt2spk | sort -u > $tgtdir/spk2utt

echo "TDT data prepare succeeded !"

