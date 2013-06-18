#!/bin/bash
renice 20 $$

# Copyright Ondrej Platek Apache 2.0
# based on copyrighted 2012 Vassil Panayotov recipe 
# at egs/voxforge/s5/run.sh(Apache 2.0)

. ./path.sh

# If you have cluster of machines running GridEngine you may want to
# change the train and decode commands in the file below
. ./cmd.sh

# Load few variables for changing the parameters of the training
. ./conf/train_conf.sh

# Copy the configuration files to exp directory.
# Write into the exp WARNINGs if reusing settings from another experiment!
local/save_check_conf.sh || exit 1;

if [ ! "$(ls -A data 2>/dev/null)" ]; then

  # local/voxforge_data_prep.sh --nspk_test ${nspk_test} ${SELECTED} || exit 1
  local/vystadial_data_prep.sh --every_n $everyN ${DATA_ROOT} || exit 1
  
  # prepare an ARPA LM and wordlist
  mkdir -p data/local
  # LEAVING it with OOV -> Allow train Kaldi for OOV model
  # cp -f ${DATA_ROOT}/arpa_trigram data/local/lm.arpa  
  # NOT ALLOWING OOV WORDS training & also in decoding
  grep -v -w OOV ${DATA_ROOT}/arpa_trigram > data/local/lm.arpa 
  echo '</s>' > data/local/vocab-full.txt
  tail -n +3 ${DATA_ROOT}/classic.v3.dct | cut -d ' ' -f 1 |\
      sort | uniq >> data/local/vocab-full.txt 
  
  # Prepare the lexicon and various phone lists
  # DISABLED Sequitor model: Pronunciations for OOV words are obtained using a pre-trained Sequitur model
  local/vystadial_prepare_dict.sh || exit 1 
  
  # Prepare data/lang and data/local/lang directories read it IO param describtion
  utils/prepare_lang.sh data/local/dict 'OOV' data/local/lang data/lang || exit 1
  
  # Prepare G.fst and data/{train,test} directories
  local/vystadial_format_data.sh || exit 1 
fi 
# end of generating data directory
  
  
###### TRAINING SETTINGS #######

# if ${MFCC_DIR} is empty then generate the content
if [ ! "$(ls -A ${MFCC_DIR} 2>/dev/null)" ]; then
  # Creating MFCC features and storing at ${MFCC_DIR} (Could be large).
  for x in train test ; do 
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $njobs \
  data/$x exp/make_mfcc/$x ${MFCC_DIR} || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x ${MFCC_DIR} || exit 1;
  done
fi


# Train monophone models on a subset of the data
utils/subset_data_dir.sh data/train $monoTrainData data/train.1k  || exit 1;
steps/train_mono.sh --nj $njobs --cmd "$train_cmd" data/train.1k data/lang exp/mono || exit 1;
  
# Monophone decoding
utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph || exit 1
# note: local/decode.sh calls the command line once for each
# test, and afterwards averages the WERs into (in this case
# exp/mono/decode/
steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  exp/mono/graph data/test exp/mono/decode
 
# Get alignments from monophone system.
steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
  $pdf $gauss data/train data/lang exp/mono_ali exp/tri1 || exit 1;
 
# decode tri1
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  exp/tri1/graph data/test exp/tri1/decode
 
# draw-tree data/lang/phones.txt exp/tri1/tree | dot -Tps -Gsize=8,10.5 | ps2pdf - tree.pdf
  
#align tri1 
steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
  
# train tri2a [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" $pdf $gauss \
  data/train data/lang exp/tri1_ali exp/tri2a || exit 1;
  
# decode tri2a
utils/mkgraph.sh data/lang_test exp/tri2a exp/tri2a/graph
steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  exp/tri2a/graph data/test exp/tri2a/decode
 
# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" $pdf $gauss \
  data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b/decode
 
# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
    --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;
  
#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --nj $njobs --cmd "$train_cmd" \
   data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj $njobs --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
steps/decode.sh --config conf/decode.config --iter 3 --nj $njobs --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3
 
# Do the same with boosting. train_mmi_boost is a number e.g. 0.05
steps/train_mmi.sh --boost ${train_mmi_boost} data/train data/lang \
   exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b${train_mmi_boost} || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj $njobs --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b${train_mmi_boost}/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj $njobs --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b${train_mmi_boost}/decode_it3 || exit 1;

# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj $njobs --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj $njobs --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3 || exit 1;


# Do LDA+MLLT+SAT, and decode.
steps/train_sat.sh $pdf $gauss data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph || exit 1;
steps/decode_fmllr.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;


# Align all data with LDA+MLLT+SAT system (tri3b)
steps/align_fmllr.sh --nj $njobs --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

# MMI on top of tri3b (i.e. LDA+MLLT+SAT+MMI)
steps/make_denlats.sh --config conf/decode.config \
   --nj $njobs --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  data/train data/lang exp/tri3b exp/tri3b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri3b_ali exp/tri3b_denlats exp/tri3b_mmi || exit 1;

steps/decode_fmllr.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
   exp/tri3b/graph data/test exp/tri3b_mmi/decode || exit 1;

# Do a decoding that uses the exp/tri3b/decode directory to get transforms from.
steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_mmi/decode2 || exit 1;


# first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj $njobs --cmd "$train_cmd" \
  250 data/train data/lang exp/tri3b_ali exp/dubm3b

 # Next, various fMMI+MMI configurations.
steps/train_mmi_fmmi.sh --learning-rate 0.0025 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/_ri3b_fmmi_b || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj $njobs --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/_ri3b_fmmi_b/decode_it$iter &
done

steps/train_mmi_fmmi.sh --learning-rate 0.001 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_c || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj $njobs --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_c/decode_it$iter &
done

# for indirect one, use twice the learning rate.
steps/train_mmi_fmmi_indirect.sh --learning-rate 0.002 --schedule "fmmi fmmi fmmi fmmi mmi mmi mmi mmi" \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_d || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj $njobs --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_d/decode_it$iter &
done

# SKIPPING this mixturing and speaker dependant settings
# You don't have to run all 3 of the below, e.g. you can just run the run_sgmm2x.sh
# local/run_sgmm.sh
# local/run_sgmm2.sh
# local/run_sgmm2x.sh
