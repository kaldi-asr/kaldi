#!/bin/bash

# Copyright 2017 Lucas Jo (Atlas Guide)
# Apache 2.0

# do this when the segmentation rule is changed
dataDir=$1
lmDir=$2

exists(){
	command -v "$1" >/dev/null 2>&1
}

# check morfessor installation 
if ! exists morfessor; then
	echo "Please, install Morfessor"
	exit 1
fi

trans=$dataDir/text
echo "Re-segment transcripts: $trans --------------------------------------------"
if [ ! -f $trans ]; then
	echo "transcription file is not found in "$dataDir
	exit 1
fi
cp $trans $trans".old"
awk '{print $1}' $trans".old" > $trans"_tmp_index"
cut -d' ' -f2- $trans".old" |\
	sed -E 's/\s+/ /g; s/^\s//g; s/\s$//g' |\
	morfessor -l $lmDir/zeroth_morfessor.seg -T - -o - \
	--output-format '{analysis} ' --output-newlines \
	--nosplit-re '[0-9\[\]\(\){}a-zA-Z&.,\-]+' \
	| paste -d" " $trans"_tmp_index" - > $trans
rm -f $trans"_tmp_index"

#transcripList=$(find $dataDir -name "*.norm.txt" -type f | sort)
#for transcript in $transcripList;
#do
#	echo "read: " $transcript
#	cat $transcript | awk '{print $1;}' > tmp
#	cat $transcript | awk '{$1="";print $0;}' | \
#	local/strip.py | \
#	#morfessor -l $lmDir/data/_lexicon_/mergedCorpus.model4.reduced -T - -o tmp2 --output-format '{analysis} ' --output-newlines
#	morfessor -l $lmDir/zeroth_morfessor.seg -T - -o tmp2 --output-format '{analysis} ' --output-newlines
#	#$lmDir/data/_lm_/seg2sentence.py tmp2 > tmp3
#
#	array=(${transcript//\./ })
#	echo "write: " ${array[0]}.${array[1]}.txt
#	paste -d" " tmp tmp2 > ${array[0]}.${array[1]}.txt
#done
#rm -f tmp*
