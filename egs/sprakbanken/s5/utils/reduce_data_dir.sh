#!/bin/bash

# koried, 10/29/2012

# Reduce a data set based on a list of turn-ids

if [ $# != 3 ]; then
echo "usage: $0 srcdir turnlist destdir"
exit 1;
fi

srcdir=$1
reclist=$2
destdir=$3

if [ ! -f $srcdir/utt2spk ]; then 
echo "$0: no such file $srcdir/utt2spk"
exit 1;
fi

function do_filtering {
# assumes the utt2spk and spk2utt files already exist.
	[ -f $srcdir/feats.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/feats.scp >$destdir/feats.scp
	[ -f $srcdir/wav.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/wav.scp >$destdir/wav.scp
	[ -f $srcdir/text ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/text >$destdir/text
	[ -f $srcdir/spk2gender ] && utils/filter_scp.pl $destdir/spk2utt <$srcdir/spk2gender >$destdir/spk2gender
	[ -f $srcdir/cmvn.scp ] && utils/filter_scp.pl $destdir/spk2utt <$srcdir/cmvn.scp >$destdir/cmvn.scp
	if [ -f $srcdir/segments ]; then
		utils/filter_scp.pl $destdir/utt2spk <$srcdir/segments >$destdir/segments
		awk '{print $2;}' $destdir/segments | sort | uniq > $destdir/reco # recordings.
		# The next line would override the command above for wav.scp, which would be incorrect.
		[ -f $srcdir/wav.scp ] && utils/filter_scp.pl $destdir/reco <$srcdir/wav.scp >$destdir/wav.scp
		[ -f $srcdir/reco2file_and_channel ] && \
			utils/filter_scp.pl $destdir/reco <$srcdir/reco2file_and_channel >$destdir/reco2file_and_channel
		
		# Filter the STM file for proper sclite scoring (this will also remove the comments lines)
		[ -f $srcdir/stm ] && utils/filter_scp.pl $destdir/reco < $srcdir/stm > $destdir/stm
		rm $destdir/reco
	fi
	srcutts=`cat $srcdir/utt2spk | wc -l`
	destutts=`cat $destdir/utt2spk | wc -l`
	echo "Reduced #utt from $srcutts to $destutts"
}

mkdir -p $destdir

# filter the utt2spk based on the set of recordings
utils/filter_scp.pl $reclist < $srcdir/utt2spk > $destdir/utt2spk

utils/utt2spk_to_spk2utt.pl < $destdir/utt2spk > $destdir/spk2utt
do_filtering;

