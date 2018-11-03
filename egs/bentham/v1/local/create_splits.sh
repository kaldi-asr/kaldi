#!/bin/bash
download_dir=$1
mkdir -p $download_dir/{train,val,test}/{pages,lines,xml,text}

partition_dir=$download_dir/"gt/Partitions/"
lines_dir=$download_dir/"gt/Images/Lines/"
pages_dir=$download_dir/"images/Images/Pages/"
xml_dir=$download_dir/"gt/PAGE/"
text_dir=$download_dir/"gt/Transcriptions/" 

function split {
	echo "Creating $1 split"
	split_dir=$download_dir/$1
	page_file=$partition_dir/$2
	line_file=$partition_dir/$3
	
	while read -r line; do
	    name="$line"
	    mv $pages_dir/$name* $split_dir/pages/  
	    mv $xml_dir/$name* $split_dir/xml/
	done < "$page_file"

	while read -r line; do
	    name="$line"
	    mv $lines_dir/$name* $split_dir/lines/
	    mv $text_dir/$name* $split_dir/text/
	done < "$line_file"
}

split train Train.lst TrainLines.lst
split val Validation.lst ValidationLines.lst
split test Test.lst TestLines.lst