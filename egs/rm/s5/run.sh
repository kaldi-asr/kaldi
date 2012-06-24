#!/bin/bash

# CAUTION: I changed e.g. 1.trans to trans.1 in the scripts.  If you ran it
# part-way through prior to this, to convert to the new naming
# convention, run:
# for x in `find . -name '*.trans'`; do mv $x `echo $x | perl -ane 's/(\d+)\.trans/trans.$1/;print;'`; done
# but be careful as this will not follow soft links.

. cmd.sh

if false; then ##TEMP

# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio

#local/rm_data_prep.sh /mnt/matylda2/data/RM || exit 1;
local/rm_data_prep.sh /export/corpora5/LDC/LDC93S3A/rm_comp || exit 1;

utils/prepare_lang.sh data/local/dict '!SIL' data/local/lang data/lang || exit 1;

local/rm_prepare_grammar.sh || exit 1;

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
featdir=mfcc

for x in test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92 train; do
  steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/$x exp/make_mfcc/$x $featdir  || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $featdir  || exit 1;
  #steps/make_plp.sh data/$x exp/make_plp/$x $featdir 4
done

# Make a combined data dir where the data from all the test sets goes-- we do
# all our testing on this averaged set.  This is just less hassle.  We
# regenerate the CMVN stats as one of the speakers appears in two of the 
# test sets; otherwise tools complain as the archive has 2 entries.
utils/combine_data.sh data/test data/test_{mar87,oct87,feb89,oct89,feb91,sep92}
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $featdir  

utils/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;

steps/train_mono.sh --nj 4 --cmd "$train_cmd" data/train.1k data/lang exp/mono  || exit 1;

#show-transitions data/lang/phones.txt exp/tri2a/final.mdl  exp/tri2a/final.occs | perl -e 'while(<>) { if (m/ sil /) { $l = <>; $l =~ m/pdf = (\d+)/|| die "bad line $l";  $tot += $1; }} print "Total silence count $tot\n";'



utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph
# note: local/decode.sh calls the command line once for each
# test, and afterwards averages the WERs into (in this case
# exp/mono/decode/
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/mono/graph data/test exp/mono/decode


# Get alignments from monophone system.
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
 1800 9000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri1/graph data/test exp/tri1/decode

#draw-tree data/lang/phones.txt exp/tri1/tree | dot -Tps -Gsize=8,10.5 | ps2pdf - tree.pdf

# align tri1
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2a [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" 1800 9000 \
 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

# decode tri2a
utils/mkgraph.sh data/lang exp/tri2a exp/tri2a/graph
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri2a/graph data/test exp/tri2a/decode

# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" 1800 9000 \
  data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b/decode

# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
   data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;

#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3

# Do the same with boosting.
steps/train_mmi.sh --boost 0.05 data/train data/lang \
   exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b0.05 || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3 || exit 1;

fi ##TEMP
# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3 || exit 1;


## Do LDA+MLLT+SAT, and decode.
steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph || exit 1;
steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;


# Align all data with LDA+MLLT+SAT system (tri3b)
steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

## MMI on top of tri3b (i.e. LDA+MLLT+SAT+MMI)
steps/make_denlats.sh --config conf/decode.config \
   --nj 8 --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  data/train data/lang exp/tri3b exp/tri3b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri3b_ali exp/tri3b_denlats exp/tri3b_mmi || exit 1;

steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
   exp/tri3b/graph data/test exp/tri3b_mmi/decode || exit 1;

# Do a decoding that uses the exp/tri3b/decode directory to get transforms from.
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_mmi/decode2 || exit 1;


#first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 8 --cmd "$train_cmd" \
  250 data/train data/lang exp/tri3b_ali exp/dubm3b

# Next, various fMMI+MMI configurations.
steps/train_mmi_fmmi.sh --learning-rate 0.0025 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_b || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_b/decode_it$iter &
done

steps/train_mmi_fmmi.sh --learning-rate 0.001 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_c || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_c/decode_it$iter &
done

# for indirect one, use twice the learning rate.
steps/train_mmi_fmmi_indirect.sh --learning-rate 0.002 --schedule "fmmi fmmi fmmi fmmi mmi mmi mmi mmi" \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_d || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_d/decode_it$iter &
done

# You don't have to run all 3 of the below, e.g. you can just run the run_sgmm2x.sh
local/run_sgmm.sh
local/run_sgmm2.sh
local/run_sgmm2x.sh

