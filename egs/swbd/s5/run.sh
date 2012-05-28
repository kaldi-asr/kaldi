#!/bin/bash

# Note:  this is a work in progress, but should run up to the point where it says:
# I AM HERE 

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. cmd.sh

# Data prep

#local/swbd_p1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2
local/swbd_p1_data_prep.sh /export/corpora3/LDC/LDC97S62 

local/swbd_p1_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

local/swbd_p1_train_lms.sh

local/swbd_p1_format_data.sh

# Data preparation and formatting for eval2000 (note: the "text" file
# is not very much preprocessed; for actual WER reporting we'll use
# sclite.
#local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/2000_hub5_eng_eval_tr
local/eval2000_data_prep.sh /export/corpora2/LDC/LDC2002S09/hub5e_00 /export/corpora2/LDC/LDC2002T43

. cmd.sh
# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfcc

steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir || exit 1;
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir || exit 1;

# after this, the next command will remove the small number of utterances
# that couldn't be extracted for some reason (e.g. too short; no such file).
utils/fix_data_dir.sh data/train || exit 1;
steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_mfcc/eval2000 $mfccdir || exit 1;
steps/compute_cmvn_stats.sh data/eval2000 exp/make_mfcc/eval2000 $mfccdir || exit 1;
utils/fix_data_dir.sh data/eval2000 # remove segments that had problems, e.g. too short.

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.

utils/subset_data_dir.sh --first data/train 4000 data/train_dev # 5.3 hours.
n=$[`cat data/train/segments | wc -l` - 4000]
utils/subset_data_dir.sh --last data/train $n data/train_nodev


# Now-- there are 264k utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.
utils/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort
utils/subset_data_dir.sh  data/train_100kshort 10000 data/train_10k
local/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

# Take the first 30k utterances (about 1/8th of the data)
utils/subset_data_dir.sh --first data/train_nodev 30000 data/train_30k
local/remove_dup_utts.sh 200 data/train_30k data/train_30k_nodup

local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup

# Take the first 100k utterances (just under half the data); we'll use
# this for later stages of training.
utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup

# The next commands are not necessary for the scripts to run, but increase 
# efficiency of data access by putting the mfcc's of the subset 
# in a contiguous place in a file.
( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_10k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_10k_nodup.ark,$mfccdir/kaldi_swbd_10k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_10k_nodup.scp data/train_10k_nodup/feats.scp
)
( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_30k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_30k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_30k_nodup.ark,$mfccdir/kaldi_swbd_30k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_30k_nodup.scp data/train_30k_nodup/feats.scp
)
 

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a || exit 1;

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup data/lang exp/mono0a_ali exp/tri1 || exit 1;

utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph

steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
  exp/tri1/graph data/eval2000 exp/tri1/decode_eval2000

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/eval2000 exp/tri2/decode_eval2000 || exit 1;
)&



steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 30k_nodup data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   2500 20000 data/train_30k_nodup data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/eval2000 exp/tri3a/decode_eval2000 || exit 1;
)&


# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri3a exp/tri3a_ali_100k_nodup || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  2500 20000 data/train_100k_nodup data/lang exp/tri3a_ali_100k_nodup exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/eval2000 exp/tri4a/decode_eval2000
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/train_dev exp/tri4a/decode_train_dev
)&

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_ali_100k_nodup


( # Build a SGMM system on just the 100k_nodup data, on top of LDA+MLLT+SAT.
 steps/train_ubm.sh --cmd "$train_cmd" 700 data/train_100k_nodup data/lang \
  exp/tri4a_ali_100k_nodup exp/ubm5a || exit 1;
 steps/train_sgmm.sh --cmd "$train_cmd" --phn-dim 50 \
   4500 40000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup \
     exp/ubm5a/final.ubm exp/sgmm5a || exit 1;

 utils/mkgraph.sh data/lang_test exp/sgmm5a exp/sgmm5a/graph || exit 1;

 steps/decode_sgmm.sh  --cmd "$decode_cmd" --config conf/decode.config \
   --nj 30 --transform-dir exp/tri4a/decode_eval2000 \
   exp/sgmm5a/graph data/eval2000 exp/sgmm5a/decode_eval2000

 # Now discriminatively train the SGMM system on 100k_nodup data.
 steps/align_sgmm.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4a_ali_100k_nodup \
   --use-graphs true --use-gselect true data/train_100k_nodup data/lang exp/sgmm5a exp/sgmm5a_ali_100k_nodup

  # Took the beam down to 10 to get acceptable decoding speed.
  steps/make_denlats_sgmm.sh --nj 30 --sub-split 30 --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
    --transform-dir exp/tri4a_ali_100k_nodup \
    data/train_100k_nodup data/lang exp/sgmm5a_ali_100k_nodup exp/sgmm5a_denlats_100k_nodup

  steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri4a_ali_100k_nodup --boost 0.1 \
    data/train_100k_nodup data/lang exp/sgmm5a_ali_100k_nodup exp/sgmm5a_denlats_100k_nodup exp/sgmm5a_mmi_b0.1

  for iter in 1 2 3 4; do
     steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4a/decode_eval2000 data/lang_test data/eval2000 exp/sgmm5a/decode_eval2000 \
       exp/sgmm5a_mmi_b0.1/decode_eval2000_it$iter &
  done

 
)

# Building a larger SAT system.

steps/train_sat.sh --cmd "$train_cmd" \
  3500 100000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri5a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config \
   --nj 30 exp/tri5a/graph data/eval2000 exp/tri5a/decode_eval2000 || exit 1;
)

# MMI starting from system in tri5a.  Use the same data (100k_nodup).
# Later we'll use all of it.
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri5a exp/tri5a_ali_100k_nodup || exit 1;
steps/make_denlats.sh --nj 40 --cmd "$decode_cmd" --transform-dir exp/tri5a_ali_100k_nodup \
   --config conf/decode.config \
   --sub-split 50 data/train_100k_nodup data/lang exp/tri5a exp/tri5a_denlats_100k_nodup  || exit 1;
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 \
  data/train_100k_nodup data/lang exp/tri5a_{ali,denlats}_100k_nodup exp/tri5a_mmi_b0.1 || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri5a/decode_eval2000 \
  exp/tri5a/graph data/eval2000 exp/tri5a_mmi_b0.1/decode_eval2000 &

steps/train_diag_ubm.sh --silence-weight 0.5 --nj 40 --cmd "$train_cmd" \
  700 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm

steps/train_mmi_fmmi.sh --learning-rate 0.005 \
  --boost 0.1 --cmd "$train_cmd" \
 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm exp/tri5a_denlats_100k_nodup \
   exp/tri5a_fmmi_b0.1 || exit 1;
 # TODO: rerun this, was done with wrong config:
 for iter in 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
     --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
     exp/tri5a/graph data/eval2000 exp/tri5a_fmmi_b0.1/decode_eval2000_it$iter &
 done

steps/train_mmi_fmmi.sh --learning-rate 0.005 \
  --boost 0.1 --cmd "$train_cmd" \
 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm exp/tri5a_denlats_100k_nodup \
   exp/tri5a_fmmi_b0.1_b || exit 1;
 for iter in 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
     --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
     exp/tri5a/graph data/eval2000 exp/tri5a_fmmi_b0.1_b/decode_eval2000_it$iter &
 done


#TEMP:
steps/train_mmi_fmmi_indirect.sh \
  --boost 0.1 --cmd "$train_cmd" \
 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm exp/tri5a_denlats_100k_nodup \
   exp/tri5a_fmmi_b0.1_indirect || exit 1;

 for iter in 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
     --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
     exp/tri5a/graph data/eval2000 exp/tri5a_fmmi_b0.1_indirect/decode_eval2000_it$iter &
 done




#HERE.
exit 0;


  # Use a smaller beam for Switchboard, as in test time.  Use the 100k dataset,
  # but include duplicates for discriminative training.
  steps/make_denlats_lda_etc.sh --nj 40 --sub-split 40 --cmd "$train_cmd" \
    $decode_opts1 --lattice-beam 6.0 data/train_100k data/lang exp/tri5a_ali_100k exp/tri5a_denlats_100k
  steps/train_lda_etc_mmi.sh --nj 40 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri5a_ali_100k exp/tri5a_denlats_100k exp/tri5a exp/tri5a_mmi
  utils/decode.sh -l data/lang_test --nj 30 --cmd "$decode_cmd" --opts "$decode_opts1" \
     steps/decode_lda_etc.sh exp/tri5a/graph data/eval2000 exp/tri5a_mmi/decode_eval2000 \
     exp/tri5a/decode_eval2000
  steps/train_lda_etc_mmi.sh --boost 0.1 --nj 40 --cmd "$train_cmd" \
    data/train_100k data/lang exp/tri5a_ali_100k exp/tri5a_denlats_100k \
    exp/tri5a exp/tri5a_mmi_b0.1
  utils/decode.sh -l data/lang_test --nj 30 --cmd "$decode_cmd" --opts "$decode_opts1" \
    steps/decode_lda_etc.sh exp/tri5a/graph  data/eval2000 exp/tri5a_mmi_b0.1/decode_eval2000 \
    exp/tri5a/decode_eval2000
)


# Align the 5a system; we'll train triphone and SGMM systems on
# all the data, on top of this.
steps/align_lda_mllt_sat.sh  --nj 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri5a exp/tri5a_ali_nodup

( # Train triphone system on all the data.
 steps/train_lda_mllt_sat.sh  --nj 30 --cmd "$train_cmd" \
   4000 150000 data/train_nodup data/lang exp/tri5a_ali_nodup exp/tri6a

 utils/mkgraph.sh data/lang_test exp/tri6a exp/tri6a/graph
 utils/decode.sh --opts "$decode_opts2" \
   -l data/lang_test --nj 30 --cmd "$decode_cmd" \
   steps/decode_lda_mllt_sat.sh exp/tri6a/graph data/eval2000 exp/tri6a/decode_eval2000

 utils/decode.sh --opts "$decode_opts2" \
   --nj 30 --cmd "$decode_cmd" \
   steps/decode_lda_mllt_sat.sh exp/tri6a/graph data/train_dev exp/tri6a/decode_train_dev

)



# getting results (see RESULTS file)
for x in exp/*/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null



