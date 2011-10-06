#!/bin/bash
#
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

# for example /mnt/matylda2/data/SWITCHBOARD_1R2

. path.sh

#check existing directories
if [ $# != 1 ]; then
   echo "Usage: ./run.sh /path/to/SWBD"
   exit 1; 
fi 

SWBD_DIR=$1
DIR=$PWD

mkdir -p data/local
cd data/local

# Audio data directory check
if [ ! -d $SWBD_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

# Trans directory check
if [ ! -d swb_ms98_transcriptions ]; then
   # To get the SWBD transcriptions and dict, do:
   echo " *** Downloading trascriptions and dictionary ***"   
   wget http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
   tar -xf switchboard_word_alignments.tar.gz
else
  echo "Directory with transcriptions exist, skipping downloading"
fi


# Option A: SWBD dictionary file check
if [ ! -f swb_ms98_transcriptions/sw-ms98-dict.text ]; then
   echo  "SWBD dictionary file does not exists"
   exit 1;
fi

# find sph audio files
(
  find $SWBD_DIR -iname '*.sph';
) > train_sph.flist

if [ `cat train_sph.flist | wc -l` -ne 2435 ]; then
  echo Warning: expected 2435 data data files, found `cat train_sph.flist | wc -l`
fi

# (1a) Transcriptions preparation
# make basic transcription file (add segments info)
awk '{name=substr($1,1,6);gsub("^sw","sw0",name); side=substr($1,7,1);stime=$2;etime=$3;
printf("%s-%s_%06.0f-%06.0f", name, side, int(100*stime+0.5), int(100*etime+0.5));
for(i=4;i<=NF;i++) printf " " toupper($i); printf "\n"}' swb_ms98_transcriptions/*/*/*-trans.text > swb.transc 

# test if trans. file is sorted
../../scripts/is_sorted.sh swb.transc

# Remove SILENCE.
# Note: we have [NOISE], [VOCALIZED-NOISE], [LAUGHTER], [SILENCE].
# removing [SILENCE] and giving phones to the other three (NSN, SPN, LAU). 
# There is also a silence phone, SIL.

cat swb.transc | perl -e 's:\b\[SILENCE]\b::g; print; ' > swb.filt.transc

#(2a) Dictionary preparation:
# Pre-processing (Upper-case, remove comments)
awk 'BEGIN{getline}($0 !~ /^#/) {$0=toupper($0); print}' swb_ms98_transcriptions/sw-ms98-dict.text | sort | awk '($0 !~ /^[:space:]*$/) {print}'> lex.text

# Get OOVs
cat swb.filt.transc | $DIR/scripts/oov2unk.pl lex.text " " 2> oovs.old.txt >/dev/null  || exit 1 #get file oovs.old.txt

# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

$DIR/scripts/dct2phones.awk lex.text | sort | \
perl -ane 's:\r::; print;' | \
awk 'BEGIN{print "<eps> 0"; print "SIL 1"; print "SPN 2"; print "NSN 3"; print "LAU 4"; N=5; } 
           {printf("%s %d\n", $1, N++); }
           {printf("%s_B %d\n", $1, N++); }
           {printf("%s_E %d\n", $1, N++); }
           {printf("%s_S %d\n", $1, N++); } ' > phones.txt


# First make a version of the lexicon without the silences etc, but with the position-markers.
# Remove the comments from the cmu lexicon and remove the (1), (2) from words with multiple 
# pronunciations.
grep -v ';;;' lex.text | perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
 | perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
   if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
     for($n=1;$n<@A-1;$n++) { print "$A[$n] "; } print "$A[$n]_E\n"; } ' \
  > lexicon_nosil.txt

# Add to cmudict the silences, noises etc.
(echo '!SIL SIL'; echo '<s> '; echo '</s> '; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - lexicon_nosil.txt  > lexicon.txt


silphones="SIL SPN NSN LAU";
# Generate colon-separated lists of silence and non-silence phones.
$DIR/scripts/silphones.pl phones.txt "$silphones" silphones.csl nonsilphones.csl

# This adds disambig symbols to the lexicon and produces data_prep/lexicon_disambig.txt

ndisambig=`$DIR/scripts/add_lex_disambig.pl lexicon.txt lexicon_disambig.txt`
echo $ndisambig > lex_ndisambig
# Next, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
$DIR/scripts/add_disambig.pl --include-zero phones.txt $ndisambig > phones_disambig.txt

# Make the words symbol-table; add the disambiguation symbol #0 (we use this in place of epsilon
# in the grammar FST).
cat lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > words.txt

# (1b) Continue trans preparation 
#Convert real OOVs to <SPOKEN_NOISE>
spoken_noise_word="<SPOKEN_NOISE>";
cat swb.filt.transc | $DIR/scripts/oov2unk.pl lexicon.txt $spoken_noise_word 2> oovs.txt | sort > train.txt || exit 1 # the .txt is the final transcript.


# (1c) Make segment files from transcript

# I) list of all segments
$DIR/scripts/make_segments.awk train.txt > segments

awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' train_sph.flist > train_sph.scp 

sph2pipe=`cd ../../../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
if [ ! -f $sph2pipe ]; then
   echo "Could not find the sph2pipe program at $sph2pipe";
   exit 1;
fi
cat train_sph.scp | awk '{printf("%s-A '$sph2pipe' -f wav -p -c 1 %s |\n", $1, $2); printf("%s-B '$sph2pipe' -f wav -p -c 2 %s |\n", $1, $2);}' | \
sort > train_wav.scp #side A - channel 1, side B - channel 2


cat segments | awk '{spk=substr($1,4,6); print $1 " " spk}' > train.utt2spk
cat train.utt2spk | sort -k 2 | $DIR/scripts/utt2spk_to_spk2utt.pl > train.spk2utt

echo SWBD_data_prep Succeeded.
