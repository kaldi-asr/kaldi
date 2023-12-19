#!/bin/bash

#Copy this in the file containing all text files you want to combine and run the script.

##Use this code from any directory (as long as target directory is properly written e.g. /home/<username>/path/to/file"
##SAGE KHAN

##Use this code from any directory (as long as target directory is properly written e.g. /home/<username>/path/to/file/ (remember to put a forward slash in the end)

target= echo "Enter target directory: "
read target

#ext= echo "Enter extension of files you want as output:" #can be used if you want to customize the output file extension
#read ex

#dest= echo "Enter Destination file name with extension: " #Also can be used for customizing output file name along with extension
#read dest

mkdir .dump
mv $target/o1.txt $target/.dump/o1-old.txt 
mv $target/o2.txt $target/.dump/o2-old.txt
mv $target/file-content-list.txt $target/.dump/output-old.txt || true #Ensure no o1,o2 and file-content-list.txt file is in target

for f in "$target"/*;
do
    echo -e $(basename "$f" '\t') >>o1.txt && echo $(cat "$f") >>o2.txt
done 
#| awk 'END { printf("File count: %d", NR); } NF=NF' ## Use this one with "done" (previous line) to get file count if needed

paste -d' ' $target/o1.txt $target/o2.txt | column -s $'\t' -t >> file-content-list.txt #Output file is printed. Remove it and from the target if you plan on reusing there.
rm $target/o1.txt 
rm $target/o2.txt






## This code is simple but there are sorting issues
#myfile = echo "Enter file name and extension you want as output: "
#read myfile
#for file in $(find . -type f -iname "*.txt"); do cat $file && printf "\n"; done >> $myfile
#find . -type f -name '*.txt' -printf '%f\t%p\n' && cat $file | sort -k1 >> file-content-list.txt






##-------------------# https://stackoverflow.com/questions/72887965/how-to-merge-multiple-text-files-and-save-them-as-csv-or-txt-without-sorting-err #
# For each txt file
#for f in "$target"/*.txt; do
   # outupt the filename name without .txt extension
#   basname "$f" .txt
   # Output the file contents with newlines replaced by a space.
#   tr '\n' ' ' <"$f"
#done |
# Join two lines of output by a tabulation. The delimiter is arbitrary.
#paste -d $'\t' - - |
# Columnate the output.
#column -s $'\t' -t
