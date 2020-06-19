#!/usr/bin/env bash
# Copyright   2018   Desh Raj (Johns Hopkins University) 

# This script reads the extracted Bentham database files and creates
#    the following files (for all the data subsets):
#    text, utt2spk, images.scp.

download_dir=$1
save_dir=$2
mkdir -p $save_dir/{train,val,test}
touch $save_dir/{train,val,test}/{text,images.scp,utt2spk,spk2utt}

partition_dir=$download_dir"/gt/Partitions/"
lines_dir=$download_dir"/gt/Images/Lines/"
text_dir=$download_dir"/gt/Transcriptions/" 

function split {
	echo "Creating $1 split"
	split_dir=$save_dir/$1
	line_file=$partition_dir/$2

	while read -r line; do
	    name="$line"
        spkid=${name:0:11}
        echo -n $name" " | cat - $text_dir/$name* >> $split_dir/text
        echo >> $split_dir/text
        echo $name $lines_dir"/"$name".png" >> $split_dir/images.scp
        echo $name $spkid >> $split_dir/utt2spk 
	done < "$line_file"
   
    perl -i -ne 'print if /\S/' $split_dir/images.scp $split_dir/text $split_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $split_dir/utt2spk > $split_dir/spk2utt
}

split train TrainLines.lst
split val ValidationLines.lst
split test TestLines.lst
