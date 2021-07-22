#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
nj=600


. ./utils/parse_options.sh

data=$1
decode_dir=$2



sdata=$data/split$nj


[ -d $data/done_fast ] && rm -rf $data/done_fast
mkdir -p $data/done_fast
[ -d $data/todo_fast ] && rm -rf $data/todo_fast
mkdir -p $data/todo_fast


for i in $(seq $nj) ; do 
	log_file=$decode_dir/log/decode.$i.log
	if [ -f $log_file ] ; then
		end_line=$(grep "with status 0" $log_file | wc -l )
		if [ $end_line -eq 1 ] ; then
			echo "$i done"
			cat $sdata/$i/text >> $data/done_fast/text
		else 
			echo "$i todo"
			cat $sdata/$i/text >> $data/todo_fast/text
		fi
	else 
		echo "$i todo"
		cat $sdata/$i/text >> $data/todo_fast/text
	fi
done

