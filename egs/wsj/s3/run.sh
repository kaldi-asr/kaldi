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

# Note: this is work in progress!  This will be the new, "cleaner" version
# of the WSJ scripts.

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# might want to run this script on a machine that has plenty of memory.


# The next command line is an example; you have to give the script
# command-line arguments corresponding to the WSJ disks from LDC.  
# Another example set of command line arguments is
# /ais/gobi2/speech/WSJ/*/??-{?,??}.?.   These must be absolute,
# not relative, pathnames.
local/wsj_data_prep.sh /mnt/matylda2/data/WSJ?/??-{?,??}.?

local/wsj_prepare_dict.sh

local/wsj_format_data.sh

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_wsj_mfcc
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
 steps/make_mfcc.sh data/$x exp/make_mfcc/$x $mfccdir 4
done


mkdir data/train_si84
for x in feats.scp text utt2spk wav.scp; do
  head -7138 data/train_si284/$x > data/train_si84/$x
done
scripts/utt2spk_to_spk2utt.pl data/train_si84/utt2spk > data/train_si84/spk2utt
scripts/filter_scp.pl data/train_si84/spk2utt data/train_si284/spk2gender > data/train_si84/spk2gender

# Now make subset with the shortest 2k utterances from si-84.
scripts/subset_data_dir.sh data/train_si84 2000 data/train_si84_2kshort

# Now make subset with half of the data from si-84.
scripts/subset_data_dir.sh data/train_si84 3500 data/train_si84_half

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"

steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang exp/mono0a

(
scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a exp/mono0a/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_dev93 exp/mono0a/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92
)&

# This queue option will be supplied to all alignment
# and training scripts.  Note: you have to supply the same num-jobs
# to the alignment and training scripts, as the archives are split
# up in this way.


steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84_half data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
    2000 10000 data/train_si84_half data/lang exp/mono0a_ali exp/tri1

scripts/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr

scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri1/graph_tgpr data/test_dev93 exp/tri1/decode_tgpr_dev93
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst \
  data/lang_test_tgpr/words.txt data/test_dev93 exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_tg

# Align tri1 system with si84 data.
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84


# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2a
scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a exp/tri2a/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri2a/graph_tgpr data/test_dev93 exp/tri2a/decode_tgpr_dev93


# Train tri2b, which is LDA+MLLT, on si84 data.
steps/train_lda_mllt.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b
scripts/mkgraph.sh data/lang_test_tgpr exp/tri2b exp/tri2b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b/decode_tgpr_eval92
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93

# Align tri2b system with si84 data.
steps/align_lda_mllt.sh  --num-jobs 10 --cmd "$train_cmd" \
  --use-graphs data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84

# Train LDA+ET system.
steps/train_lda_et.sh --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2c
scripts/mkgraph.sh data/lang_test_tgpr exp/tri2c exp/tri2c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_et.sh exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_et_2pass.sh exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_tgpr_dev93_2pass

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_lda_mllt_sat.sh  --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b
scripts/mkgraph.sh data/lang_test_tgpr exp/tri3b exp/tri3b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri3b/graph_tgpr \
  data/test_dev93 exp/tri3b/decode_tgpr_dev93
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst \
  data/lang_test_tgpr/words.txt data/test_dev93 exp/tri3b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93_tg
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri3b/graph_tgpr \
  data/test_eval92 exp/tri3b/decode_tgpr_eval92
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst \
  data/lang_test_tgpr/words.txt data/test_eval92 exp/tri3b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92_tg

# From 3b system, align all si284 data.
steps/align_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284

steps/train_lda_etc_quick.sh --num-jobs 10 --cmd "$train_cmd" \
   4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284 exp/tri4b
scripts/mkgraph.sh data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b/decode_tgpr_dev93

# Train UBM, for SGMM system on top of LDA+MLLT.
steps/train_ubm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c
steps/align_lda_mllt.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84_10
steps/train_sgmm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
   3500 10000 41 40 data/train_si84 data/lang exp/tri2b_ali_si84_10 exp/ubm3c/final.ubm exp/sgmm3c
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm3c exp/sgmm3c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh exp/sgmm3c/graph_tgpr data/test_dev93 exp/sgmm3c/decode_tgpr_dev93
 

# Train SGMM system on top of LDA+MLLT+SAT.
steps/align_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri3b exp/tri3b_ali_si84
steps/train_ubm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri3b_ali_si84 exp/ubm4b
steps/train_sgmm_lda_etc.sh  --num-jobs 10 --cmd "$train_cmd" \
  3500 10000 41 40 data/train_si84 data/lang exp/tri3b_ali_si84 exp/ubm4b/final.ubm exp/sgmm4b
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm4b exp/sgmm4b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_dev93 exp/sgmm4b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92


# Align 3b system with si284 data and num-jobs = 20; we'll train an LDA+MLLT+SAT system on si284 from this.
# This is 4c.  c.f. 4b which is "quick" training.

steps/align_lda_mllt_sat.sh --num-jobs 20 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284_20
steps/train_lda_mllt_sat.sh --num-jobs 20 --cmd "$train_cmd" \
  4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/tri4c
scripts/mkgraph.sh data/lang_test_tgpr exp/tri4c exp/tri4c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri4c/graph_tgpr \
  data/test_dev93 exp/tri4c/decode_tgpr_dev93

# Train SGMM on top of LDA+MLLT+SAT, on all SI-284 data.  C.f. 4b which was
# just on SI-84.
steps/train_ubm_lda_etc.sh --num-jobs 20 --cmd "$train_cmd" \
  600 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/ubm4c
steps/train_sgmm_lda_etc.sh  --num-jobs 20 --cmd "$train_cmd" \
  5500 25000 50 40 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/ubm4c/final.ubm exp/sgmm4c
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm4c exp/sgmm4c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4c/graph_tgpr data/test_dev93 exp/sgmm4c/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst \
  data/lang_test_tgpr/words.txt data/test_dev93 exp/sgmm4c/decode_tgpr_dev93 exp/sgmm4c/decode_tgpr_dev93_tg

# decode the above with nov'92 too
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4c/graph_tgpr data/test_eval92 exp/sgmm4c/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst \
  data/lang_test_tgpr/words.txt data/test_eval92 exp/sgmm4c/decode_tgpr_eval92 exp/sgmm4c/decode_tgpr_eval92_tg




############# END  ###################



# exp/decode_mono_tgpr_eval92 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/test_eval92.scp 

# add --no-queue --num-jobs 4 after "scripts/decode.sh" below, if you don't have
# qsub (i.e. Sun Grid Engine) on your system.  The number of jobs to use depends
# on how many CPUs and how much memory you have, on the local machine.  If you do
# have qsub on your system, you will probably have to edit steps/decode.sh anyway
# to change the queue options... or if you have a different queueing system,
# you'd have to modify the script to use that.

(scripts/mkgraph.sh --mono data/G_tg_pruned.fst exp/mono/tree exp/mono/final.mdl exp/graph_mono_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_mono_tgpr_eval92 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_mono_tgpr_eval93 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/test_eval93.scp 
) &


##########
# 
# Old run.sh is from here [parts of it not yet incorporated]


silphones="SIL SPN NSN";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/phones.txt "$silphones" data/silphones.csl data/nonsilphones.csl

# This adds disambig symbols to the lexicon and produces data/lexicon_disambig.txt

ndisambig=`scripts/add_lex_disambig.pl data/lexicon.txt data/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
echo $ndisambig > data/lex_ndisambig
# Next, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
scripts/add_disambig.pl --include-zero data/phones.txt $ndisambig > data/phones_disambig.txt


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

for x in test_eval92 test_dev93 test_eval93; do 
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

scripts/make_lexicon_fst.pl data/lexicon_disambig.txt 0.5 SIL '#'$ndisambig | \
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
fstdeterminizestar data/L_disambig.fst >/dev/null  



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
 scripts/decode.sh exp/decode_mono_tgpr_eval92 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_mono_tgpr_eval93 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/test_eval93.scp 
) &

steps/train_tri1.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri1/tree exp/tri1/final.mdl exp/graph_tri1_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri1_tgpr_eval92 exp/graph_tri1_tg_pruned/HCLG.fst steps/decode_tri1.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_tri1_tgpr_eval93 exp/graph_tri1_tg_pruned/HCLG.fst steps/decode_tri1.sh data/test_eval93.scp 
) &

steps/train_tri2a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2a/tree exp/tri2a/final.mdl exp/graph_tri2a_tg_pruned || exit 1;
  for year in 92 93; do
   scripts/decode.sh exp/decode_tri2a_tgpr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a.sh data/test_eval${year}.scp 
   scripts/decode.sh exp/decode_tri2a_tgpr_fmllr_utt_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_fmllr.sh data/test_eval${year}.scp 
   scripts/decode.sh exp/decode_tri2a_tgpr_dfmllr_utt_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_dfmllr.sh data/test_eval${year}.scp 
   scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_fmllr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_fmllr.sh data/test_eval${year}.scp 
   scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_dfmllr_fmllr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_dfmllr_fmllr.sh data/test_eval${year}.scp 
   scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_dfmllr_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_dfmllr.sh data/test_eval${year}.scp 
 done

)&

# also doing tri2a with bigram [+ lattice generation + rescoring]
(
 scripts/mkgraph.sh data/G_bg.fst exp/tri2a/tree exp/tri2a/final.mdl exp/graph_tri2a_bg || exit 1;
 for year in 92 93; do
  scripts/decode.sh exp/decode_tri2a_bg_eval${year} exp/graph_tri2a_bg/HCLG.fst steps/decode_tri2a.sh data/test_eval${year}.scp 
  scripts/decode.sh exp/decode_tri2a_bg_latgen_eval${year} exp/graph_tri2a_bg/HCLG.fst steps/decode_tri2a_latgen.sh data/test_eval${year}.scp 
  scripts/latoracle.sh exp/decode_tri2a_bg_latgen_eval${year} data/test_eval${year}.txt exp/decode_tri2a_bg_latoracle_eval${year}
  scripts/latrescore.sh exp/decode_tri2a_bg_latgen_eval${year} data/G_bg.fst data/G_tg.fst data/test_eval${year}.txt exp/decode_tri2a_bg_rescore_tg_eval${year} 
  scripts/latrescore.sh exp/decode_tri2a_bg_latgen_eval${year} data/G_bg.fst data/G_tg_pruned.fst data/test_eval${year}.txt exp/decode_tri2a_bg_rescore_tg_pruned_eval${year} 
  scripts/latrescore.sh exp/decode_tri2a_bg_latgen_eval${year} data/G_bg.fst data/G_bg.fst data/test_eval${year}.txt exp/decode_tri2a_bg_rescore_bg_eval${year} 
 done

 for year in 92 93; do
  scripts/decode.sh exp/decode_tri2a_bg_latgen_beam15_eval${year} exp/graph_tri2a_bg/HCLG.fst steps/decode_tri2a_latgen_beam15.sh data/test_eval${year}.scp 
  scripts/decode.sh exp/decode_tri2a_tgpr_beam15_eval${year} exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_beam15.sh data/test_eval${year}.scp 

  scripts/latrescore.sh exp/decode_tri2a_bg_latgen_beam15_eval${year} data/G_bg.fst data/G_tg.fst data/test_eval${year}.txt exp/decode_tri2a_bg15_rescore_tg_eval${year} 
  scripts/latrescore.sh exp/decode_tri2a_bg_latgen_beam15_eval${year} data/G_bg.fst data/G_tg_pruned.fst data/test_eval${year}.txt exp/decode_tri2a_bg15_rescore_tg_pruned_eval${year} 
  scripts/latrescore.sh exp/decode_tri2a_bg_latgen_beam15_eval${year} data/G_bg.fst data/G_bg.fst data/test_eval${year}.txt exp/decode_tri2a_bg15_rescore_bg_eval${year} 
 done
 )&




steps/train_tri3a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri3a/tree exp/tri3a/final.mdl exp/graph_tri3a_tg_pruned || exit 1;
for year in 92 93; do
 scripts/decode.sh exp/decode_tri3a_tgpr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a.sh data/test_eval${year}.scp 
# per-speaker fMLLR
scripts/decode.sh --per-spk exp/decode_tri3a_tgpr_fmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_fmllr.sh data/test_eval${year}.scp
# per-utterance fMLLR
scripts/decode.sh exp/decode_tri3a_tgpr_uttfmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_fmllr.sh data/test_eval${year}.scp 
# per-speaker diagonal fMLLR
scripts/decode.sh --per-spk exp/decode_tri3a_tgpr_dfmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_diag_fmllr.sh data/test_eval${year}.scp 
# per-utterance diagonal fMLLR
scripts/decode.sh exp/decode_tri3a_tgpr_uttdfmllr_eval${year} exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_diag_fmllr.sh data/test_eval${year}.scp 
done
)&

# also doing tri3a with bigram
(
 scripts/mkgraph.sh data/G_bg.fst exp/tri3a/tree exp/tri3a/final.mdl exp/graph_tri3a_bg || exit 1;
 scripts/decode.sh exp/decode_tri3a_bg_eval92 exp/graph_tri3a_bg/HCLG.fst steps/decode_tri3a.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_tri3a_bg_eval93 exp/graph_tri3a_bg/HCLG.fst steps/decode_tri3a.sh data/test_eval93.scp 
 scripts/decode.sh exp/decode_tri3a_bg_latgen_eval92 exp/graph_tri3a_bg/HCLG.fst steps/decode_tri3a_latgen.sh data/test_eval92.scp 
 scripts/latrescore.sh exp/decode_tri3a_bg_latgen_eval92 data/G_bg.fst data/G_tg.fst data/test_eval92.txt exp/decode_tri3a_bg_rescore_tg_eval92 
)&


#### Now alternative experiments... ###

# Exponential Transform (ET)
steps/train_tri2b.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2b/tree exp/tri2b/final.mdl exp/graph_tri2b_tg_pruned || exit 1;
  for year in 92 93; do
 scripts/decode.sh exp/decode_tri2b_tgpr_utt_eval${year} exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b.sh data/test_eval${year}.scp 
 scripts/decode.sh --per-spk exp/decode_tri2b_tgpr_eval${year} exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b.sh data/test_eval${year}.scp 
 scripts/decode.sh exp/decode_tri2b_tgpr_utt_fmllr_eval${year} exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b_fmllr.sh data/test_eval${year}.scp 
 scripts/decode.sh --per-spk exp/decode_tri2b_tgpr_fmllr_eval${year} exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b_fmllr.sh data/test_eval${year}.scp 
done

) &

# Cepstral Mean Normalization (CMN)
steps/train_tri2c.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2c/tree exp/tri2c/final.mdl exp/graph_tri2c_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2c_tgpr_utt_eval92 exp/graph_tri2c_tg_pruned/HCLG.fst steps/decode_tri2c.sh data/test_eval92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2c_tgpr_eval92 exp/graph_tri2c_tg_pruned/HCLG.fst steps/decode_tri2c.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_tri2c_tgpr_utt_eval93 exp/graph_tri2c_tg_pruned/HCLG.fst steps/decode_tri2c.sh data/test_eval93.scp 
 scripts/decode.sh --per-spk exp/decode_tri2c_tgpr_eval93 exp/graph_tri2c_tg_pruned/HCLG.fst steps/decode_tri2c.sh data/test_eval93.scp 
)&


# MLLT/STC
steps/train_tri2d.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2d/tree exp/tri2d/final.mdl exp/graph_tri2d_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2d_tgpr_eval92 exp/graph_tri2d_tg_pruned/HCLG.fst steps/decode_tri2d.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_tri2d_tgpr_eval93 exp/graph_tri2d_tg_pruned/HCLG.fst steps/decode_tri2d.sh data/test_eval93.scp 
 )&

# Splice+LDA
steps/train_tri2e.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2e/tree exp/tri2e/final.mdl exp/graph_tri2e_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2e_tgpr_eval92 exp/graph_tri2e_tg_pruned/HCLG.fst steps/decode_tri2e.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_tri2e_tgpr_eval93 exp/graph_tri2e_tg_pruned/HCLG.fst steps/decode_tri2e.sh data/test_eval93.scp 
 )&

# Splice+LDA+MLLT
steps/train_tri2f.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2f/tree exp/tri2f/final.mdl exp/graph_tri2f_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2f_tgpr_eval92 exp/graph_tri2f_tg_pruned/HCLG.fst steps/decode_tri2f.sh data/test_eval92.scp  
 scripts/decode.sh exp/decode_tri2f_tgpr_eval93 exp/graph_tri2f_tg_pruned/HCLG.fst steps/decode_tri2f.sh data/test_eval93.scp  
)&

# Linear VTLN (+ regular VTLN)
steps/train_tri2g.sh
(
 scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2g/tree exp/tri2g/final.mdl exp/graph_tri2g_tg_pruned || exit 1;

for year in 92 93; do
 scripts/decode.sh exp/decode_tri2g_tgpr_utt_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g.sh data/test_eval${year}.scp  
 scripts/decode.sh exp/decode_tri2g_tgpr_utt_diag_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_diag.sh data/test_eval${year}.scp  
 scripts/decode.sh --wav exp/decode_tri2g_tgpr_utt_vtln_diag_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_vtln_diag.sh data/test_eval${year}.scp  
 scripts/decode.sh --per-spk exp/decode_tri2g_tgpr_diag_fmllr_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_diag_fmllr.sh data/test_eval${year}.scp  
 scripts/decode.sh --per-spk exp/decode_tri2g_tgpr_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g.sh data/test_eval${year}.scp  
 scripts/decode.sh --per-spk exp/decode_tri2g_tgpr_diag_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_diag.sh data/test_eval${year}.scp  
 scripts/decode.sh --wav --per-spk exp/decode_tri2g_tgpr_vtln_diag_eval${year} exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_vtln_diag.sh data/test_eval${year}.scp  
done

)&

# Splice+HLDA
steps/train_tri2h.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2h/tree exp/tri2h/final.mdl exp/graph_tri2h_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2h_tgpr_eval92 exp/graph_tri2h_tg_pruned/HCLG.fst steps/decode_tri2h.sh data/test_eval92.scp  
 scripts/decode.sh exp/decode_tri2h_tgpr_eval93 exp/graph_tri2h_tg_pruned/HCLG.fst steps/decode_tri2h.sh data/test_eval93.scp  
)&

# Triple-deltas + HLDA
steps/train_tri2i.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2i/tree exp/tri2i/final.mdl exp/graph_tri2i_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2i_tgpr_eval92 exp/graph_tri2i_tg_pruned/HCLG.fst steps/decode_tri2i.sh data/test_eval92.scp  
 scripts/decode.sh exp/decode_tri2i_tgpr_eval93 exp/graph_tri2i_tg_pruned/HCLG.fst steps/decode_tri2i.sh data/test_eval93.scp  
)&

# Splice + HLDA
steps/train_tri2j.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2j/tree exp/tri2j/final.mdl exp/graph_tri2j_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2j_tgpr_eval92 exp/graph_tri2j_tg_pruned/HCLG.fst steps/decode_tri2j.sh data/test_eval92.scp
 scripts/decode.sh exp/decode_tri2j_tgpr_eval93 exp/graph_tri2j_tg_pruned/HCLG.fst steps/decode_tri2j.sh data/test_eval93.scp 
 )&


# LDA+ET
steps/train_tri2k.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2k/tree exp/tri2k/final.mdl exp/graph_tri2k_tg_pruned || exit 1;
 for year in 92 93; do
  scripts/decode.sh exp/decode_tri2k_tgpr_utt_eval$year exp/graph_tri2k_tg_pruned/HCLG.fst steps/decode_tri2k.sh data/test_eval$year.scp 
  scripts/decode.sh --per-spk exp/decode_tri2k_tgpr_eval$year exp/graph_tri2k_tg_pruned/HCLG.fst steps/decode_tri2k.sh data/test_eval$year.scp 
  scripts/decode.sh --per-spk exp/decode_tri2k_tgpr_fmllr_eval$year exp/graph_tri2k_tg_pruned/HCLG.fst steps/decode_tri2k_fmllr.sh data/test_eval$year.scp 
 done
 )&

# the same on all the data..
steps/train_tri3k.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri3k/tree exp/tri3k/final.mdl exp/graph_tri3k_tg_pruned || exit 1;
 for year in 92 93; do
  scripts/decode.sh exp/decode_tri3k_tgpr_utt_eval$year exp/graph_tri3k_tg_pruned/HCLG.fst steps/decode_tri3k.sh data/test_eval$year.scp 
  scripts/decode.sh --per-spk exp/decode_tri3k_tgpr_eval$year exp/graph_tri3k_tg_pruned/HCLG.fst steps/decode_tri3k.sh data/test_eval$year.scp 
  scripts/decode.sh --per-spk exp/decode_tri3k_tgpr_fmllr_eval$year exp/graph_tri3k_tg_pruned/HCLG.fst steps/decode_tri3k_fmllr.sh data/test_eval$year.scp 
 done
 )&

# LDA+MLLT+SAT
steps/train_tri2l.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2l/tree exp/tri2l/final.mdl exp/graph_tri2l_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2l_tgpr_utt_eval92 exp/graph_tri2l_tg_pruned/HCLG.fst steps/decode_tri2l.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_tri2l_tgpr_utt_eval93 exp/graph_tri2l_tg_pruned/HCLG.fst steps/decode_tri2l.sh data/test_eval93.scp 
 scripts/decode.sh --per-spk exp/decode_tri2l_tgpr_eval92 exp/graph_tri2l_tg_pruned/HCLG.fst steps/decode_tri2l.sh data/test_eval92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2l_tgpr_eval93 exp/graph_tri2l_tg_pruned/HCLG.fst steps/decode_tri2l.sh data/test_eval93.scp 
 )&



# LDA + MLLT + Linear VTLN (+ regular VTLN)
# Note: this depends on tri2f.
steps/train_tri2m.sh
(
 scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2m/tree exp/tri2m/final.mdl exp/graph_tri2m_tg_pruned || exit 1;

for year in 92 93; do
 scripts/decode.sh exp/decode_tri2m_tgpr_utt_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m.sh data/test_eval${year}.scp  
 scripts/decode.sh exp/decode_tri2m_tgpr_utt_diag_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m_diag.sh data/test_eval${year}.scp  
 scripts/decode.sh --wav exp/decode_tri2m_tgpr_utt_vtln_diag_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m_vtln_diag.sh data/test_eval${year}.scp  

 scripts/decode.sh --per-spk exp/decode_tri2m_tgpr_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m.sh data/test_eval${year}.scp 
 scripts/decode.sh --per-spk exp/decode_tri2m_tgpr_diag_fmllr_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m_diag_fmllr.sh data/test_eval${year}.scp  
 scripts/decode.sh --per-spk exp/decode_tri2m_tgpr_diag_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m_diag.sh data/test_eval${year}.scp  
 scripts/decode.sh --wav --per-spk exp/decode_tri2m_tgpr_vtln_diag_eval${year} exp/graph_tri2m_tg_pruned/HCLG.fst steps/decode_tri2m_vtln_diag.sh data/test_eval${year}.scp  
done

)&


train_ubm2a.sh || exit 1;

# Deltas + SGMM
steps/train_sgmm2a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm2a/tree exp/sgmm2a/final.mdl exp/graph_sgmm2a_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_sgmm2a_tgpr_eval92 exp/graph_sgmm2a_tg_pruned/HCLG.fst steps/decode_sgmm2a.sh data/test_eval92.scp 
 scripts/decode.sh exp/decode_sgmm2a_tgpr_eval93 exp/graph_sgmm2a_tg_pruned/HCLG.fst steps/decode_sgmm2a.sh data/test_eval93.scp )&

# + speaker vectors
steps/train_sgmm2b.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm2b/tree exp/sgmm2b/final.mdl exp/graph_sgmm2b_tg_pruned || exit 1;
 for year in 92 93; do
  scripts/decode.sh --per-spk exp/decode_sgmm2b_tgpr_eval${year} exp/graph_sgmm2b_tg_pruned/HCLG.fst steps/decode_sgmm2b.sh data/test_eval${year}.scp 
  scripts/decode.sh exp/decode_sgmm2b_tgpr_utt_eval${year} exp/graph_sgmm2b_tg_pruned/HCLG.fst steps/decode_sgmm2b.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk  exp/decode_sgmm2b_fmllr_tgpr_eval${year} exp/graph_sgmm2b_tg_pruned/HCLG.fst steps/decode_sgmm2b_fmllr.sh data/test_eval${year}.scp 
 done
)&

# as sgmm2b, but with LDA+STC.  Depends on tri2f.
# Note: increased acwt from 12 to 13.
steps/train_ubm2c.sh || exit 1;
steps/train_sgmm2d.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm2d/tree exp/sgmm2d/final.mdl exp/graph_sgmm2d_tg_pruned || exit 1;
 for year in 92 93; do
  scripts/decode.sh --per-spk exp/decode_sgmm2d_tgpr_eval${year} exp/graph_sgmm2d_tg_pruned/HCLG.fst steps/decode_sgmm2d.sh data/test_eval${year}.scp 
  scripts/decode.sh exp/decode_sgmm2d_tgpr_utt_eval${year} exp/graph_sgmm2d_tg_pruned/HCLG.fst steps/decode_sgmm2d.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk  exp/decode_sgmm2d_fmllr_tgpr_eval${year} exp/graph_sgmm2d_tg_pruned/HCLG.fst steps/decode_sgmm2d_fmllr.sh data/test_eval${year}.scp 
 done
)&

# Note: sgmm2e depends on tri2k
steps/train_ubm2d.sh || exit 1;
steps/train_sgmm2e.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm2e/tree exp/sgmm2e/final.mdl exp/graph_sgmm2e_tg_pruned || exit 1;
 for year in 92 93; do
  scripts/decode.sh --per-spk exp/decode_sgmm2e_tgpr_eval${year} exp/graph_sgmm2e_tg_pruned/HCLG.fst steps/decode_sgmm2e.sh data/test_eval${year}.scp exp/graph_tri2k_tg_pruned/HCLG.fst
  scripts/decode.sh exp/decode_sgmm2e_tgpr_utt_eval${year} exp/graph_sgmm2e_tg_pruned/HCLG.fst steps/decode_sgmm2e.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk  exp/decode_sgmm2e_fmllr_tgpr_eval${year} exp/graph_sgmm2e_tg_pruned/HCLG.fst steps/decode_sgmm2e_fmllr.sh data/test_eval${year}.scp 
 done
)&


# [on all the data]
steps/train_ubm3a.sh || exit 1;
steps/train_sgmm3b.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm3b/tree exp/sgmm3b/final.mdl exp/graph_sgmm3b_tg_pruned || exit 1;
 for year in 92 93; do 
  scripts/decode.sh --per-spk exp/decode_sgmm3b_tgpr_eval${year} exp/graph_sgmm3b_tg_pruned/HCLG.fst steps/decode_sgmm3b.sh data/test_eval${year}.scp 
  scripts/decode.sh exp/decode_sgmm3b_tgpr_utt_eval${year} exp/graph_sgmm3b_tg_pruned/HCLG.fst steps/decode_sgmm3b.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk  exp/decode_sgmm3b_fmllr_tgpr_eval${year} exp/graph_sgmm3b_tg_pruned/HCLG.fst steps/decode_sgmm3b_fmllr.sh data/test_eval${year}.scp 
 done
)&

# [ gender dependent ]
steps/train_ubm3b.sh || exit 1;
steps/train_sgmm3c.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm3c/tree exp/sgmm3c/final.mdl exp/graph_sgmm3c_tg_pruned || exit 1;
 for year in 92 93; do
  scripts/decode.sh --per-spk exp/decode_sgmm3c_tgpr_eval${year} exp/graph_sgmm3c_tg_pruned/HCLG.fst steps/decode_sgmm3c.sh data/test_eval${year}.scp 
  scripts/decode.sh exp/decode_sgmm3c_tgpr_utt_eval${year} exp/graph_sgmm3c_tg_pruned/HCLG.fst steps/decode_sgmm3c.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk  exp/decode_sgmm3c_fmllr_tgpr_eval${year} exp/graph_sgmm3c_tg_pruned/HCLG.fst steps/decode_sgmm3c_fmllr.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk exp/decode_sgmm3c_tgpr_norm_eval${year} exp/graph_sgmm3c_tg_pruned/HCLG.fst steps/decode_sgmm3c_norm.sh data/test_eval${year}.scp 
  scripts/decode.sh --per-spk  exp/decode_sgmm3c_fmllr_tgpr_norm_eval${year} exp/graph_sgmm3c_tg_pruned/HCLG.fst steps/decode_sgmm3c_fmllr_norm.sh data/test_eval${year}.scp
 done
)&


# [ with ET features]
steps/train_ubm3d.sh || exit 1;
steps/train_sgmm3e.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/sgmm3e/tree exp/sgmm3e/final.mdl exp/graph_sgmm3e_tg_pruned || exit 1;
for year in 92 93; do
  scripts/decode.sh --per-spk exp/decode_sgmm3e_tgpr_eval${year} exp/graph_sgmm3e_tg_pruned/HCLG.fst steps/decode_sgmm3e.sh data/test_eval${year}.scp exp/graph_tri2k_tg_pruned/HCLG.fst
  scripts/decode.sh exp/decode_sgmm3e_tgpr_utt_eval${year} exp/graph_sgmm3e_tg_pruned/HCLG.fst steps/decode_sgmm3e.sh data/test_eval${year}.scp exp/graph_tri2k_tg_pruned/HCLG.fst
  scripts/decode.sh --per-spk  exp/decode_sgmm3e_fmllr_tgpr_eval${year} exp/graph_sgmm3e_tg_pruned/HCLG.fst steps/decode_sgmm3e_fmllr.sh data/test_eval${year}.scp exp/graph_tri2k_tg_pruned/HCLG.fst
done
)&



# see RESULTS for results...

# For an e.g. of scoring with sclite: do e.g.
# scripts/score_sclite.sh exp/decode_tri2a_tgpr_eval92 data/test_eval92.txt
# cat exp/decode_tri2a_tgpr_eval92/scoring/hyp.sys


# notes on timing of alignment... trying it in tri2a for 500 utts.  Took
# [with retry-beam=40]
# the results below seem to show that beam = 6 is the fastest...
# of course this assumes the retry-beam is 40.
# 20.9 sec @ beam = 7
# 13.8 sec @ beam = 6
# 14.4 sec @ beam = 5
# 14.4 sec @ beam = 4

#How I moved stuff to log/ dir:
#for x in train_lda_mllt.sh train_deltas.sh; do cat $x | perl -ane 's:dir/(\S+)\.log:dir/log/$1.log:; print; ' | sed 's:mkdir -p $dir:mkdir -p $dir/log:' > tmpf; cp tmpf $x; done