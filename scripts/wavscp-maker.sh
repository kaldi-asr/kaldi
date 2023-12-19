#!/bin/bash

#This program prints out list of all files in the directory along with its extension. Its a one click solution for creating wav.scp files 
##Use this code from any directory (as long as target directory is properly written e.g. /home/<username>/path/to/file"

##SAGE KHAN

target= echo "Enter target directory (write full path e.g. /home/abc/Documents/....): "
read target

#ext= echo "Enter extension: "
#read ext

find "$target" -type f -name '*.wav' -printf '%f\t%p\n' | sort -k1 >> wav-tmp.scp



### For a single folder
#cd $target

#mkdir $target/.dump
#mv $target/tmp-wav.txt $target/.dump/tmp-wav-old.txt
#mv $target/wav.tmp.scp $target/.dump/wav-tmp-old.scp
#mv $target/wav.scp $target/.dump/wav-old.scp
#mv $target/o1.txt $target/.dump/o1.txt

#for f in "$target"/*
#do
    #printf "$f\n" >> wav-tmp.scp      
#    cat $f >> tmp-wav.txt &&
#    cut -c tmp-wav.txt | paste -d , wav-tmp.scp -
#    echo -e $(basename "$f" '\t') >>o1.txt  
#done

#paste -d' ' $target/o1.txt $target/wav-tmp.scp | column -s $'\t' -t >> wav.scp

#rm $target/o1.txt
#rm $target/wav-tmp.scp
#rm $target/tmp-wav.txt
########################

##list all files in a directory and subdirectory linux
#find . -type f -follow -print  #EG#find "$target" -type f -name "*.$ext" >>wav-tmp.scp

#for file in $(find . -type f -iname "*.txt"); do cat $file && printf "\n"; done >> $myfile
#find . -type f -name '*.txt' -printf '%f\t%p\n' && cat $file | sort -k1 >> file-content-list.txt

#dir list all files in subdirectories
#dir *.txt *.doc		# filter by extension (both doc and txt)
#dir	/a:-d			# files only (no subfolders)
#dir /s				# current directory and subfolders content
#dir /s /a:-d		# files only (including subfolders)
#dir > myfile.txt	# stored in myfile.txt (dir /s > myfile.txt with subfolders)
#dir /o:[sortorder] 	# example:  dir /o:-s    (sort by decreasing size)
#  N : By name (alphabetic).
#  S : By size (smallest first).
#  E : By extension (alphabetic).
#  D : By date/time (oldest first).
#  - : Prefix to reverse order.
  
##find . -type f -name "*.txt"  
  
######Bash-specific
#####bash list all files in directory and subdirectories
#### syntax 
#ls -R <file-or-directory-to-find>

#### example
#ls -R *hotographi*


##https://linuxhint.com/find-all-files-using-extension-linux/
