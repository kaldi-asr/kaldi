#!/bin/bash

echo "#### BATCH AUDIO FILE CONVERTER (USING LIST OF AUDIO FILE NAMES SAVED IN A COLUMN IN TXT FILE) ####"

target= echo "Enter target directory to save output files (write full path e.g. /home/abc/Documents/....): "
read target

filename= echo "Enter file containing list of wav files (write full path e.g. /home/abc/Documents/....): "
read filename

#cmd= echo "Enter Command you want to run line by line (use "$line" as input variable for each line and "$n" to add a number for renaming): "
#read cmd   ###If used, would require lots of explanation echoed on shell 

n=0  ###Activate if you want to save in numbered form
#mkdir $target/output


#while read -r line; do
#### Reading each line
    #n=$((n+1)) ###Activate if you want to save in numbered form
#    base= echo $(basename $line)
#    read base
#    dir= echo $(dirname $line)
#    read dir     
#    sox "$line" -r 16000 -b 16 -c 1 "$dir/$base"
#done < $filename

for line in `cat $filename`;
do
    n=$((n+1)) ###Activate if you want to save in numbered form
    base= basename $line
    read base
    dir= readlink -f $line
    read dir
    path= $dir/$base
    read path     
    echo "Processing file $line"
    cd $dir
    sox $base -r 16000 -b 16 -c 1 '$target/$base' 

done < $filename


################################################################
#input="/path/to/txt/file"
#while IFS= read -r line
#do
#  echo "$line"
#done < "$input"



################################################################
#n=1
#while read line; do
###### reading each line
#echo "Line No. $n : $line"
#n=$((n+1))
#done < $filename

################################################################
#while IFS= read -r line; do printf '%s\n' "$line"; done < input_file

################################################################
#while read -r line; do
#### Reading each line
#echo $line
#done < company2.txt

################################################################
#i=0;
#for filename in /home/mrityunjoy/myWork/audio_files/*.wav; do
#    i=$((i+1));
#    sox "$filename" -r 16000 -b 16 -c 1 "file$i.wav"
#done

################################################################
#a=0;
#for i in `ls *.wav`;
#do

#let a++;
#echo "Processing file $i"
#sox $i -r 16000 -b 16 -c 1 file$a.wav 

#done


#find ./ -name "*wav" -exec sox {} -r 16000 -b 16 -c 1 {}.16000.wav \;

