#!/bin/bash

#Copy this in the file containing all text files you want to combine and run the script.
#This will convert audio files in a folder to required format 
##Use this code from any directory (as long as target directory is properly written e.g. /home/<username>/path/to/file"
##SAGE KHAN


target= echo "Enter target directory: "
read target

extin= echo "enter extension of input audio file: "
read extin

extout= echo "enter extension of output audio file: "
read extout

outch= echo "enter output channel: "
read outch

#dest= echo "Enter Destination file name with extension: " #Also can be used for customizing output file name along with extension
#read dest

for f in "$target"/*."$extin";
do
    sox --channels "$outch" "$f"."$extin" "$f"-converted."$extout"    
done | awk 'END { printf("File count: %d", NR); } NF=NF' 


##SLICER command
#ffmpeg -i "$f".wav -ss $start -to $end -c copy "$f"-converted.wav ##time format is 00:00:01 


#In command syntax, the effects step is, confusingly, written last. That means the pipeline is composed this way:

#input → combine → output → effects

#The simplest conversion command involves only an input file and an output file. Here's the command to convert an MP3 file to a lossless FLAC file:

# sox countdown.mp3 output.flac ##Check outputusing soxi command

#sox intro.ogg intro.flac fade p 3 8 ##This applies a three-second fade-in to the head of the audio and a fade-out starting at the eight-second mark (the intro music is only 11 seconds, so the fade-out is also three-seconds in this case):

# sox intro.ogg output.flac gain -1 stretch 1.35 fade p 0 6 ##This command applies a -1 gain effect, a tempo stretch of 1.35, and a fade-out:
# sox countdown.mp3 intro.ogg output.flac ##In this example, output.flac now contains countdown audio, followed immediately by intro music.


#The effects chain is specified at the end of a command. It can alter audio prior to sending the data to its final destination. For instance, sometimes audio that's too loud can cause problems during conversion:'

# sox bad.wav bad.ogg
#sox WARN sox: `bad.ogg' output clipped 126 samples; decrease volume?

##Applying a gain effect can often solve this problem:

# sox bad.wav bad.ogg gain -1

#INTEGRATE MORE OF SOX VIA http://sox.sourceforge.net/sox.html

##sox 833676000239153.wav 8-conv2.wav rate 16000 gain -1

## This code is simple but there are sorting issues
#myfile = echo "Enter file name and extension you want as output: "
#read myfile
#for file in $(find . -type f -iname "*.txt"); do cat $file && printf "\n"; done >> $myfile

#find . -type f -name '*.txt' -printf '%f\t%p\n' && cat $file | sort -k1 >> file-content-list.txt


#sox full_length.wav trimmed.wav fade 0 -5 0.01 ##Parameter 1 is '0' so there is no fade in. Parameter 2 removes the last 5 seconds Parameter 3 uses a 10ms fade
