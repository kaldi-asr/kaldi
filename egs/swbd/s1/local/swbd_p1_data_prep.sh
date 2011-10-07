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
  echo "Directory with transcriptions exists, skipping downloading"
fi


# Option A: SWBD dictionary file check
if [ ! -f swb_ms98_transcriptions/sw-ms98-dict.text ]; then
   echo  "SWBD dictionary file does not exist"
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
 for(i=4;i<=NF;i++) printf " " toupper($i); printf "\n"}' \
  swb_ms98_transcriptions/*/*/*-trans.text  > transcripts1.txt

# test if trans. file is sorted
../../scripts/is_sorted.sh transcripts1.txt

# Remove SILENCE.
# Note: we have [NOISE], [VOCALIZED-NOISE], [LAUGHTER], [SILENCE].
# removing [SILENCE], and the <B_ASIDE> and <E_ASIDE> markers that mark
# speech to somone; we will give phones to the other three (NSN, SPN, LAU). 
# There will also be a silence phone, SIL.

cat transcripts1.txt | perl -ane 's:\s\[SILENCE\](\s|$):$1:g; s/<B_ASIDE>//g; s/<E_ASIDE>//g; print; ' | \
  awk '{if(NF > 1) { print; } } ' > transcripts2.txt

#(2a) Dictionary preparation:
# Pre-processing (Upper-case, remove comments)
awk 'BEGIN{getline}($0 !~ /^#/) {$0=toupper($0); print}' \
  swb_ms98_transcriptions/sw-ms98-dict.text | sort | awk '($0 !~ /^[:space:]*$/) {print}' \
   > lexicon1.txt

# Get OOV list.
# oovs.old.txt is used just for debugging.  should be empty except for
# [NOISE], [VOCALIZED-NOISE], [LAUGHTER]
cat transcripts1.txt | $DIR/scripts/oov2unk.pl lexicon1.txt " " \
   2> oovs.old.txt >/dev/null  || exit 1 #get file oovs.old.txt



# Now modify both the lexicon and trancsripts to



# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.


$DIR/local/dct2phones.awk lexicon1.txt | sort | \
  perl -ane 's:\r::; print;' | \
  awk 'BEGIN{print "<eps> 0"; print "SIL 1"; print "SPN 2"; print "NSN 3"; print "LAU 4"; N=5; } 
           {printf("%s %d\n", $1, N++); }
           {printf("%s_B %d\n", $1, N++); }
           {printf("%s_E %d\n", $1, N++); }
           {printf("%s_S %d\n", $1, N++); } ' > phones.txt


# First make a version of the lexicon without the silences etc, but with the position-markers.
# Remove the comments from the lexicon and remove the (1), (2) from words with multiple 
# pronunciations.
cat lexicon1.txt \
 | perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
   if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
     for($n=1;$n<@A-1;$n++) { print "$A[$n] "; } print "$A[$n]_E\n"; } ' \
  > lexicon2.txt

# Add to the lexicon the silences, noises etc.
(echo '!SIL SIL'; echo '[VOCALIZED-NOISE] SPN'; echo '[NOISE] NSN'; echo '[LAUGHTER] LAU' ) | \
 cat - lexicon2.txt  > lexicon3.txt


# Get lexicon mapping.  That is-- for each word in the lexicon, we map it
# to a new written form.  The transformations we do are:
# remove laughter markings, e.g.
# [LAUGHTER-STORY] -> STORY
# Remove partial-words, e.g.
# -[40]1K W AH N K EY
# becomes -1K
# and
# -[AN]Y IY
# becomes
# -Y
# -[A]B[OUT]- B
# becomes
# -B-
# Also, curly braces, which appear to be used for "nonstandard"
# words or non-words, are removed, e.g. 
# {WOLMANIZED} W OW L M AX N AY Z D
# -> WOLMANIZED
# Also, mispronounced words, e.g.
#  [YEAM/YEAH] Y AE M
# are changed to just e.g. YEAM, i.e. the orthography
# of the mispronounced version.
# Note-- this is only really to be used in training.  The main practical
# reason is to avoid having tons of disambiguation symbols, which
# we otherwise would get because there are many partial words with
# the same phone sequences (most problematic: S).
# Also, map
# THEM_1 EH M -> THEM
# so that multiple pronunciations just have alternate entries
# in the lexicon.

cat lexicon3.txt | awk '{print $1}' | \
  $DIR/local/word_map.pl | sort | uniq > word_map;

# Now apply word map to the transcriptions and lexicon.
$DIR/scripts/compose_maps.pl transcripts2.txt word_map >transcripts3.txt

cp transcripts3.txt train.txt # This is the final transcripts...

cat lexicon3.txt | perl -e 'open(W, "<word_map")||die;
  while(<W>){ @A=split; $map{$A[0]} = $A[1]; }
  while(<>) { @A=split; $A[0] = $map{$A[0]}; print join(" ", @A) . "\n"; } ' \
  | sort | uniq > lexicon4.txt

cp lexicon4.txt lexicon.txt # This is the final lexicon.

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

## (1b) Continue trans preparation 
## Convert real OOVs to <SPOKEN_NOISE>

# (1c) Make segment files from transcript

# I) list of all segments
$DIR/local/make_segments.awk train.txt > segments

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

echo Switchboard phase 1 data preparation succeeded.
