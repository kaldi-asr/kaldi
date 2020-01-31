#!/usr/bin/env bash

# Copyright 2014   University of Tehran (Author: Bagher BabaAli)
# Apache 2.0.

# This script normalizes the TIMIT phonetic transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by 
# the transcript, e.g.:
#Normalizes phonetic transcriptions for TIMIT, by mapping the phones to a 
#smaller set defined by the -m option. This script assumes that the mapping is 
#done in the \"standard\" fashion, i.e. to 48 or 39 phones.  The input is 
#assumed to have 60 phones (+1 for glottal stop, which is deleted), but that can
#be changed using the -from option. The input format is assumed to be utterance 
#ID followed by transcript on the same line.


if [ $# -ne 1 ]; then
   echo "Argument should be a transcription file in a format where each line contains an utterance ID followed by the transcript."
   exit 1;
fi

cat $1 | awk  '{
                for(i=1; i<=NF; ++i) {
                  if ( $i == "\\" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "p" ) ) {
                        printf("p ")
                     }                    
                  } 
                  else if ( $i == "`" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "b" ) ) {
                        printf("b ")
                     }                    
                  } 
                  else if ( $i == "-" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "t" ) ) {
                        printf("t ")
                     }                    
                  } 
                  else if ( $i == "=" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "d" ) ) {
                        printf("d ")
                     }                    
                  } 
                  else if ( $i == "@" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "c" ) ) {
                        printf("c ")
                     }                    
                  } 
                  else if ( $i == "*" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "k" ) ) {
                        printf("k ")
                     }                    
                  } 
                  else if ( $i == "!" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != ";" ) ) {
                        printf("; ")
                     }                    
                  } 
                  else if ( $i == "&" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "g" ) ) {
                        printf("g ")
                     }                    
                  } 
                  else if ( $i == "^" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "q" ) ) {
                        printf("q ")
                     }                    
                  } 
                  else if ( $i == "#" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "," ) ) {
                        printf(", ")
                     }                    
                  } 
                  else if ( $i == "$" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "'\''" ) ) {
                        printf("'\'' ")
                     }                    
                  } 
                  else if ( $i == "(" ) {
                     if ( ( i+1 == NF ) || ( $(i+1) != "]" ) ) {
                        printf("] ")
                     }                    
                  } 
                  else {
                     printf("%s ",$i)
                  }
                }
                printf("\n");
             }' | tr 'c' 'k' | tr ';' 'g' | sed -r 's/j/sil/g' 
