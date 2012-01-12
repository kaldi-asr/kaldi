#!/bin/bash

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

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# might want to run this script on a machine that has plenty of memory.

# (1) To get the CMU dictionary, do:
svn co https://cmusphinx.svn.sourceforge.net/svnroot/cmusphinx/trunk/cmudict/
# got this at revision 10966 in the last tests done before releasing v1.0.
# can add -r 10966 for strict compatibility.


#(2) Dictionary preparation:

mkdir -p data

# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

cat cmudict/cmudict.0.7a.symbols | perl -ane 's:\r::; print;' | \
 awk 'BEGIN{print "<eps> 0"; print "SIL 1"; print "SPN 2"; print "NSN 3"; N=4; } 
           {printf("%s %d\n", $1, N++); }
           {printf("%s_B %d\n", $1, N++); }
           {printf("%s_E %d\n", $1, N++); }
           {printf("%s_S %d\n", $1, N++); } ' >data/phones.txt


# First make a version of the lexicon without the silences etc, but with the position-markers.
# Remove the comments from the cmu lexicon and remove the (1), (2) from words with multiple 
# pronunciations.

grep -v ';;;' cmudict/cmudict.0.7a | perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
 | perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
   if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
     for($n=1;$n<@A-1;$n++) { print "$A[$n] "; } print "$A[$n]_E\n"; } ' \
  > data/lexicon_nosil.txt

# Add to cmudict the silences, noises etc.

(echo '!SIL SIL'; echo '<s> '; echo '</s> '; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - data/lexicon_nosil.txt  > data/lexicon.txt


silphones="SIL SPN NSN";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/phones.txt "$silphones" data/silphones.csl data/nonsilphones.csl

# This adds disambig symbols to the lexicon and produces data/lexicon_disambig.txt

ndisambig=`scripts/add_lex_disambig.pl data/lexicon.txt data/lexicon_disambig.txt`
echo $ndisambig > data/lex_ndisambig
# Next, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
scripts/add_disambig.pl --include-zero data/phones.txt $ndisambig > data/phones_disambig.txt

# Make the words symbol-table; add the disambiguation symbol #0 (we use this in place of epsilon
# in the grammar FST).
cat data/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > data/words.txt


#(3)
# data preparation (this step requires the WSJ disks, from LDC).
# It takes as arguments a list of the directories ending in
# e.g. 11-13.1 (we don't assume a single root dir because
# there are different ways of unpacking them).

cd data_prep


# The following command needs a list of directory names from
# the LDC's WSJ disks.  These will end in e.g. 11-1.1.
# examples:
# /ais/gobi2/speech/WSJ/*/??-{?,??}.?
# /mnt/matylda2/data/WSJ?/??-{?,??}.?
./run.sh [list-of-directory-names]


cd ..



# Here is where we select what data to train on.
# use all the si284 data.
cp data_prep/train_si284_wav.scp data/train_wav.scp
cp data_prep/train_si284.txt data/train.txt
cp data_prep/train_si284.spk2utt data/train.spk2utt 
cp data_prep/train_si284.utt2spk data/train.utt2spk
cp data_prep/spk2gender.map data/

for x in eval_nov92 dev_nov93 eval_nov93; do 
  cp data_prep/$x.spk2utt data/
  cp data_prep/$x.utt2spk data/
  cp data_prep/$x.txt data/
  cp data_prep/${x}_wav.scp data/
done

cat data/train.txt | scripts/sym2int.pl --ignore-first-field data/words.txt  > data/train.tra


# Get the right paths on our system by sourcing the following shell file
# (edit it if it's not right for your setup). 
. path.sh

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
scripts/make_lexicon_fst.pl data/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=data/phones.txt --osymbols=data/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/L.fst

# Create the lexicon FST with disambiguation symbols.  There is an extra
# step where we create a loop "pass through" the disambiguation symbols
# from G.fst.  

phone_disambig_symbol=`grep \#0 data/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/words.txt | awk '{print $2}'`

scripts/make_lexicon_fst.pl data/lexicon_disambig.txt 0.5 SIL  | \
   fstcompile --isymbols=data/phones_disambig.txt --osymbols=data/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > data/L_disambig.fst


# Making the grammar FSTs 
# This step is quite specific to this WSJ setup.
# see data_prep/run.sh for more about where these LMs came from.

steps/make_lm_fsts.sh

## Sanity check; just making sure the next command does not crash. 
fstdeterminizestar data/G_bg.fst >/dev/null  

## Sanity check; just making sure the next command does not crash. 
fsttablecompose data/L_disambig.fst data/G_bg.fst | fstdeterminizestar >/dev/null


# At this point, make sure that "./exp/" is somewhere you can write
# a reasonably large amount of data (i.e. on a fast and large 
# disk somewhere).  It can be a soft link if necessary.


# (4) feature generation


# Make the training features.
# note that this runs 3-4 times faster if you compile with DEBUGLEVEL=0
# (this turns on optimization).

# Set "dir" to someplace you can write to.
#e.g.: dir=/mnt/matylda6/jhu09/qpovey/kaldi_wsj_mfcc_f
dir=[some directory to put MFCCs]
steps/make_mfcc_train.sh $dir
steps/make_mfcc_test.sh $dir

# (5) running the training and testing steps..

steps/train_mono.sh || exit 1;

# add --no-queue --num-jobs 4 after "scripts/decode.sh" below, if you don't have
# qsub (i.e. Sun Grid Engine) on your system.  The number of jobs to use depends
# on how many CPUs and how much memory you have, on the local machine.  If you do
# have qsub on your system, you will probably have to edit steps/decode.sh anyway
# to change the queue options... or if you have a different queueing system,
# you'd have to modify the script to use that.

(scripts/mkgraph.sh --mono data/G_tg_pruned.fst exp/mono/tree exp/mono/final.mdl exp/graph_mono_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_mono_tgpr_eval92 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/eval_nov92.scp 
 scripts/decode.sh exp/decode_mono_tgpr_eval93 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/eval_nov93.scp 
) &



###### TRAIN THE BASELINE SYSTEM TO PRODUCE MLP TRAINING ALIGNMENTS #######

steps/train_tri1.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri1/tree exp/tri1/final.mdl exp/graph_tri1_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri1_tgpr_eval92 exp/graph_tri1_tg_pruned/HCLG.fst steps/decode_tri1.sh data/eval_nov92.scp 
 scripts/decode.sh exp/decode_tri1_tgpr_eval93 exp/graph_tri1_tg_pruned/HCLG.fst steps/decode_tri1.sh data/eval_nov93.scp 
) &

steps/train_tri2a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2a/tree exp/tri2a/final.mdl exp/graph_tri2a_tg_pruned || exit 1;
  for year in 92 93; do
   scripts/decode.sh exp/decode_tri2a_tgpr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a.sh data/eval_nov${year}.scp 
#   scripts/decode.sh exp/decode_tri2a_tgpr_fmllr_utt_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_fmllr.sh data/eval_nov${year}.scp 
#   scripts/decode.sh exp/decode_tri2a_tgpr_dfmllr_utt_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_dfmllr.sh data/eval_nov${year}.scp 
#   scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_fmllr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_fmllr.sh data/eval_nov${year}.scp 
#   scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_dfmllr_fmllr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_dfmllr_fmllr.sh data/eval_nov${year}.scp 
#   scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_dfmllr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_dfmllr.sh data/eval_nov${year}.scp 
 done

)&

# also doing tri2a with bigram
#(
# scripts/mkgraph.sh data/G_bg.fst exp/tri2a/tree exp/tri2a/final.mdl exp/graph_tri2a_bg || exit 1;
# for year in 92 93; do
#  scripts/decode.sh exp/decode_tri2a_bg_eval${year} exp/graph_tri2a_bg/HCLG.fst steps/decode_tri2a.sh data/eval_nov${year}.scp 
# done
#)&

steps/train_tri3a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri3a/tree exp/tri3a/final.mdl exp/graph_tri3a_tg_pruned || exit 1;
for year in 92 93; do
 scripts/decode.sh exp/decode_tri3a_tgpr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a.sh data/eval_nov${year}.scp 
# per-speaker fMLLR
#scripts/decode.sh --per-spk exp/decode_tri3a_tgpr_fmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_fmllr.sh data/eval_nov${year}.scp
# per-utterance fMLLR
#scripts/decode.sh exp/decode_tri3a_tgpr_uttfmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_fmllr.sh data/eval_nov${year}.scp 
# per-speaker diagonal fMLLR
#scripts/decode.sh --per-spk exp/decode_tri3a_tgpr_dfmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_diag_fmllr.sh data/eval_nov${year}.scp 
# per-utterance diagonal fMLLR
#scripts/decode.sh exp/decode_tri3a_tgpr_uttdfmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_diag_fmllr.sh data/eval_nov${year}.scp 
done
)&

# also doing tri3a with bigram
#(
# scripts/mkgraph.sh data/G_bg.fst exp/tri3a/tree exp/tri3a/final.mdl exp/graph_tri3a_bg || exit 1;
# scripts/decode.sh exp/decode_tri3a_bg_eval92 exp/graph_tri3a_bg/HCLG.fst steps/decode_tri3a.sh data/eval_nov92.scp 
# scripts/decode.sh exp/decode_tri3a_bg_eval93 exp/graph_tri3a_bg/HCLG.fst steps/decode_tri3a.sh data/eval_nov93.scp 
#)&




###### REDUCE NUMBER OF PDFS IN THE ALIGNMENTS TO ~500 ######
steps/reduce_pdf_count.sh




###### TRAIN MLP WITH tri3a ALIGNMENTS ######
###### reduced number of PDFs 
time scripts/run_sge_or_locally.sh "-l gpu=1 -q long.q@@pco203" steps/train_nnet_tri3a_s1.sh $PWD/exp/nnet_tri3a_s1

# decode with pruned trigram
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/reduce_pdf_count/tree exp/reduce_pdf_count/final.mdl exp/graph_nnet_tri3a_s1_tg_pruned || exit 1;
for year in 92 93; do
 scripts/decode.sh exp/decode_nnet_tri3a_s1_tgpr_eval${year} exp/graph_nnet_tri3a_s1_tg_pruned/HCLG.fst steps/decode_nnet_tri3a_s1.sh data/eval_nov${year}.scp 
done
#tune ACWT
for year in 92 93; do
 scripts/tune_acscale.py 0.0 0.5 exp/decode_nnet_tri3a_s1_tgpr_eval${year}_tune steps/decode_nnet_tri3a_s1.sh exp/graph_nnet_tri3a_s1_tg_pruned/HCLG.fst data/eval_nov${year}.scp
done
)&
# also doing nnet_tri3a_s1 with bigram
#(
# scripts/mkgraph.sh data/G_bg.fst exp/reduce_pdf_count/tree exp/reduce_pdf_count/final.mdl exp/graph_nnet_tri3a_s1_bg || exit 1;
# scripts/decode.sh exp/decode_nnet_tri3a_s1_bg_eval92 exp/graph_nnet_tri3a_s1_bg/HCLG.fst steps/decode_nnet_tri3a_s1.sh data/eval_nov92.scp 
# scripts/decode.sh exp/decode_nnet_tri3a_s1_bg_eval93 exp/graph_nnet_tri3a_s1_bg/HCLG.fst steps/decode_nnet_tri3a_s1.sh data/eval_nov93.scp 
#)&





###### TRAIN MLP WITH tri3a ALIGNMENTS ######
###### full number of PDFs (3349)
time scripts/run_sge_or_locally.sh "-l gpu=1 -q long.q@@pco203" steps/train_nnet_tri3a_s2.sh $PWD/exp/nnet_tri3a_s2

# decode with pruned trigram
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri3a/tree exp/tri3a/final.mdl exp/graph_nnet_tri3a_s2_tg_pruned || exit 1;
for year in 92 93; do
 scripts/decode.sh exp/decode_nnet_tri3a_s2_tgpr_eval${year} exp/graph_nnet_tri3a_s2_tg_pruned/HCLG.fst steps/decode_nnet_tri3a_s2.sh data/eval_nov${year}.scp 
done
#tune ACWT
for year in 92 93; do
 scripts/tune_acscale.py 0.0 0.5 exp/decode_nnet_tri3a_s2_tgpr_eval${year}_tune steps/decode_nnet_tri3a_s2.sh exp/graph_nnet_tri3a_s2_tg_pruned/HCLG.fst data/eval_nov${year}.scp
done
)&




###### TRAIN TANDEM SYSTEM WITH tri3a ALIGNMENTS ######
###### full number of PDFs (3349)

# train the MLP
time scripts/run_sge_or_locally.sh "-l gpu=1 -q long.q@@pco203" steps/train_nnet-bn_tri3a_s4b.net.sh $PWD/exp/nnet-bn_tri3a_s4b_net
# train the GMMs with LDA+MLLT
steps/train_nnet-bn_tri3a_s4b_tri2j.gmm.sh
# decode with pruned trigram
(GRAPHDIR=exp/graph_nnet-bn_tri3a_s4b_tri2j_tg_pruned/
  if [ ! -s $GRAPHDIR/HCLG.fst ]; then 
    scripts/mkgraph.sh \
      data/G_tg_pruned.fst \
      exp/nnet-bn_tri3a_s4b_tri2j.gmm/tree \
      exp/nnet-bn_tri3a_s4b_tri2j.gmm/final.mdl \
      $GRAPHDIR \
      || exit 1; 
  fi
  for year in 92 93; do
   scripts/decode.sh \
     exp/decode_nnet-bn_tri3a_s4b_tri2j_tgpr_eval${year} \
     $GRAPHDIR/HCLG.fst \
     steps/decode_nnet-bn_tri3a_s4b_tri2j.tandem.sh \
     data/eval_nov${year}.scp &
  done
)





