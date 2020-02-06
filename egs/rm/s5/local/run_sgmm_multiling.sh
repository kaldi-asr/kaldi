#!/usr/bin/env bash

# Multilingual setup for SGMMs.
# Caution: this is just a stub, intended to show some others what to do, it
# is not functional yet.

# We treat the WSJ setup as the "other language"-- in fact it's the same language,
# of course, but we treat the phones there as a distinct set.
# The only important thing is that the WSJ data has the same sample rate as the
# RM data.

# add the prefix to all the words and phones:

mkdir -p data_ml exp_ml # ml stands for "multilingual"

utils/add_lang_prefix.sh data/lang rm: data_ml/lang_rm

utils/add_lang_prefix.sh ../../wsj/s5/data/lang wsj: data_ml/lang_wsj

# add the prefix to all the words, utterance-ids, and speaker-ids.
utils/add_data_prefix.sh data/train rm: data_ml/train_rm

utils/add_data_prefix.sh ../../wsj/s5/data/train_si284 wsj: data_ml/train_si284_wsj


# Merge the "lang" directories.  This will change the phones.txt and words.txt,
# to incorporate all the symbols in the original setups.
utils/merge_lang.sh data_ml/lang_rm data_ml/lang_wsj data_ml/lang_rm_wsj

utils/combine_data.sh data_ml/train_rm data_ml/train_si284_wsj data_ml/train

# the call to utils/convert_models.sh below will
# convert the RM LDA+MLLT system to use the new "lang" directory.
# This script converts the models in the directory to use the new integer values
# for the phones, as in data/lang_rm_wsj.
# Everything else will be copied.  The only thing changed in the models is
# the transition-ids.  We'll need a program call like
#  gmm-convert <phone-map-file> <model-in> <model-out>
# where each line of phone-map-file has two lines, (phone-in phone-out).
# This will just affect the transition model, by mapping all the phone-ids.
# We'll also need a program
#  convert-tree <phone-map-file> <tree-in> <tree-out>

utils/convert_models.sh exp/tri2b data_ml/lang_rm exp_ml/tri2b_rm data_ml/lang_rm_wsj

utils/convert_models.sh ../../wsj/exp/tri4b data_ml/lang_wsj exp_ml/tri4b_wsj data_ml/lang

# Re-do the alignment of the RM tri2b setup with the converted models
# (this avoids the hassle of converting the alignment.)
steps/align_si.sh --nj 8 --cmd "$train_cmd"  data_ml/train_rm data_ml/lang exp_ml/tri2b_rm \
    exp_ml/tri2b_rm_ali || exit 1;

# Now, starting from those alignments train an RM system with the same LDA+MLLT
# matrix as the WSJ system.  The training script takes this from the alignment directory,
# so it's sufficient to put it there:
cp exp_ml/tri4b_wsj/final.mat exp_ml/tri2b_rm_ali/final.mat

steps/train_sat.sh 1800 9000 data/train data/lang exp_ml/tri2b_rm_ali exp_ml/tri3b_rm_ali || exit 1;

# Train an LDA+MLLT+SAT system for RM that uses the same LDA+MLLT transforms as for WSJ.
steps/train_sat.sh 1800 9000 data_ml/train_rm data_ml/lang exp_ml/tri2b_rm_ali exp_ml/tri3b_rm || exit 1;

# Now merge the RM and WSJ models.  This will create trees and transition-models
# that handle the two (disjoint) sets of phones that the RM and WSJ models
# contain.  We'll need a program "merge-tree" and a program "gmm-merge".  The
# "merge-tree" program will need, for each tree, a record of which sets of
# phones it was supposed to handle, since this is not recorded in the tree
# itself-- we can get this from the transition models which do record this.
# probably the "merge-tree" program will have usage:
# merge-tree <tree1> <phone-set-1> <tree2> <phone-set-2> ... <tree-out>
# where the phone-set-n's will probably be filenames that contain lists of
# the phones.
# The "gmm-merge" program will have the usage:
# gmm-merge <model1> <model2> ... <model-out>

steps/merge_models.sh data_ml/tri3b_rm data_ml/tri4b_wsj data_ml/tri4b

steps/align_fmllr.sh --nj 32 --cmd "$train_cmd" data_ml/train data_ml/lang exp_ml/tri4b \
  exp_ml/tri4b_ali || exit 1;


steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 600 \
   data_ml/train data_ml/lang exp_ml/tri4b_ali exp_ml/ubm5a || exit 1;

# Use slightly larger SGMM parameters than the WSJ setup.
steps/train_sgmm2.sh --cmd "$train_cmd" \
  15000 30000 data_ml/train data_ml/lang exp_ml/tri4b_ali \
    exp_ml/ubm5a/final.ubm exp_ml/sgmm2_5a || exit 1;


# This convert_models.sh script will also have the effect of subsetting
# the model, because some of the phones are undefined in the destination.
# We should make sure that the programs "gmm-convert" and "convert-tree"
# accept a phone map that does not map all of the phones we have-- it would
# delete those phones.  The --reduce option to the script would be passed
# into those programs, and confirm to them that that's "really" what we want
# to do.
utils/convert_models.sh --reduce true exp_ml/sgmm2_5a data_ml/lang exp/sgmm2_5c_ml data/lang

(
  utils/mkgraph.sh data/lang_test_tgpr exp/sgmm2_5c_ml exp/sgmm2_5c_ml/graph_tgpr
  steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
    exp/sgmm2_5c_ml/graph_tgpr data/test_dev93 exp/sgmm2_5c_ml/decode_tgpr_dev93
  steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_eval92 \
    exp/sgmm2_5c_ml/graph_tgpr data/test_eval92 exp/sgmm2_5c_ml/decode_tgpr_eval92

  utils/mkgraph.sh data/lang_test_bd_tgpr exp/sgmm2_5c_ml exp/sgmm2_5c_ml/graph_bd_tgpr || exit 1;
  steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
    exp/sgmm2_5c_ml/graph_bd_tgpr data/test_dev93 exp/sgmm2_5c_ml/decode_bd_tgpr_dev93
  steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
    exp/sgmm2_5c_ml/graph_bd_tgpr data/test_eval92 exp/sgmm2_5c_ml/decode_bd_tgpr_eval92
) &
