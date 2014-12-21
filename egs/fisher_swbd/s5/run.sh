#!/bin/bash

# It's best to run the commands in this one by one.

. ./cmd.sh
. ./path.sh
mfccdir=mfcc
set -e

# prepare fisher data and put it under data/train_fisher
local/fisher_data_prep.sh /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
   /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13

# at BUT:
####local/fisher_data_prep.sh /mnt/matylda6/jhu09/qpovey/FISHER/LDC2005T19 /mnt/matylda2/data/FISHER/

# prepare swbd data and put it under data/train_swbd
local/swbd1_data_prep.sh /export/corpora3/LDC/LDC97S62
# local/swbd1_data_prep.sh /data/corpora0/LDC97S62
# local/swbd1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2
# local/swbd1_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/switchboard1

# The following script prepares dictionary files for both Fisher and Switchboard (strict superset of the latter)
local/fisher_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

# LM for swbd could be used for decoding purposes
#fisher_opt="--fisher /scail/group/deeplearning/speech/datasets/LDC2004T19-Fisher-Transcripts"
#local/swbd1_train_lms.sh $fisher_opt \
#  data/local/train_swbd/text data/local/dict/lexicon.txt data/local/lm

# merge two datasets into one
mkdir -p data/train_all
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cat data/train_fisher/$f data/train_swbd/$f > data/train_all/$f
done

# LM for train_all
local/fisher_train_lms.sh 
local/fisher_create_test_lang.sh

# Prepare Eval2000 and RT-03 test sets

#local/eval2000_data_prep.sh /scail/group/deeplearning/speech/datasets/LDC2002S09/hub5e_00/ /scail/group/deeplearning/speech/datasets/LDC2002T43 || exit 1
local/eval2000_data_prep.sh /export/corpora/LDC/LDC2002S09/hub5e_00 /export/corpora/LDC/LDC2002T43 || exit 1
 
#local/rt03_data_prep.sh /scail/group/deeplearning/speech/datasets/rt_03 || exit 1
local/rt03_data_prep.sh /export/corpora/LDC/LDC2007S10 || exit 1

utils/fix_data_dir.sh data/train_all


# Make MFCCs for the training set
steps/make_mfcc.sh --nj 100 --cmd "$train_cmd" data/train_all exp/make_mfcc/train_all $mfccdir || exit 1;
utils/fix_data_dir.sh data/train_all
utils/validate_data_dir.sh data/train_all
steps/compute_cmvn_stats.sh data/train_all exp/make_mfcc/train_all $mfccdir

# subset swbd features and put them back into train_swbd in case separate training is needed
awk -F , '{print $1}' data/train_swbd/spk2utt > data/swbd_spklist
utils/subset_data_dir.sh --spk-list data/swbd_spklist data/train_all data/train_swbd
steps/compute_cmvn_stats.sh data/train_swbd exp/make_mfcc/train_all $mfccdir

# Make MFCCs for the test sets
utils/fix_data_dir.sh data/eval2000
steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/eval2000 exp/make_mfcc/eval2000 $mfccdir || exit 1;
steps/compute_cmvn_stats.sh data/eval2000 exp/make_mfcc/eval2000 $mfccdir

utils/fix_data_dir.sh data/rt03
steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/rt03 exp/make_mfcc/rt03 $mfccdir || exit 1;
steps/compute_cmvn_stats.sh data/rt03 exp/make_mfcc/rt03 $mfccdir

utils/fix_data_dir.sh data/eval2000
utils/validate_data_dir.sh data/eval2000

utils/fix_data_dir.sh data/rt03
utils/validate_data_dir.sh data/rt03

n=$[`cat data/train_all/segments | wc -l`]
utils/subset_data_dir.sh --last data/train_all $n data/train

# Now-- there are 2.1 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those. We also take these subsets from Switchboard, which has
# more carefully hand-labeled alignments

utils/subset_data_dir.sh --shortest data/train_swbd 100000 data/train_100kshort
local/remove_dup_utts.sh 10 data/train_100kshort data/train_100kshort_nodup
utils/subset_data_dir.sh  data/train_100kshort_nodup 10000 data/train_10k_nodup

utils/subset_data_dir.sh --speakers data/train_swbd 30000 data/train_30k
utils/subset_data_dir.sh --speakers data/train_swbd 100000 data/train_100k

local/remove_dup_utts.sh 200 data/train_30k data/train_30k_nodup
local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup
local/remove_dup_utts.sh 300 data/train data/train_nodup

# The next commands are not necessary for the scripts to run, but increase 
# efficiency of data access by putting the mfcc's of the subset 
# in a contiguous place in a file.
( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_10k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_fish_10k_nodup.ark,$mfccdir/kaldi_fish_10k_nodup.scp \
  && cp $mfccdir/kaldi_fish_10k_nodup.scp data/train_10k_nodup/feats.scp
)
( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_30k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_30k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_fish_30k_nodup.ark,$mfccdir/kaldi_fish_30k_nodup.scp \
  && cp $mfccdir/kaldi_fish_30k_nodup.scp data/train_30k_nodup/feats.scp
)
( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_100k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_100k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_fish_100k_nodup.ark,$mfccdir/kaldi_fish_100k_nodup.scp \
  && cp $mfccdir/kaldi_fish_100k_nodup.scp data/train_100k_nodup/feats.scp
)

# Start training on the Switchboard subset, which has cleaner alignments

steps/train_mono.sh --nj 3 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a 

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    3200 30000 data/train_30k_nodup data/lang exp/mono0a_ali exp/tri1a || exit 1;
#used to be 2500 20000
(utils/mkgraph.sh data/lang_test exp/tri1a exp/tri1a/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1a/graph data/eval2000 exp/tri1a/decode_dev
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1a/graph data/rt03 exp/tri1a/decode_rt03)&

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/tri1a exp/tri1a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    3200 30000 data/train_30k_nodup data/lang exp/tri1a_ali exp/tri1b || exit 1;
#used to be 2500 20000

(utils/mkgraph.sh data/lang_test exp/tri1b exp/tri1b/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1b/graph data/eval2000 exp/tri1b/decode_dev
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1b/graph data/rt03 exp/tri1b/decode_rt03)&

steps/align_si.sh --nj 50 --cmd "$train_cmd" \
   data/train_100k_nodup data/lang exp/tri1b exp/tri1b_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    5500 90000 data/train_100k_nodup data/lang exp/tri1b_ali exp/tri2 || exit 1;
 #used to be 2500 20000 on 30k
(  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/eval2000 exp/tri2/decode_dev || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/rt03 exp/tri2/decode_rt03 || exit 1;
)&

# Train tri3a, the last speaker-independent triphone stage, 
# on the whole Switchboard training set
steps/align_si.sh --nj 100 --cmd "$train_cmd" \
   data/train_swbd data/lang exp/tri2 exp/tri2_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    11500 200000 data/train_swbd data/lang exp/tri2_ali exp/tri3a || exit 1;
 #used to be 2500 20000
(  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/eval2000 exp/tri3a/decode_dev || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/rt03 exp/tri3a/decode_rt03 || exit 1;
)&

# Train tri3b, which is LDA+MLLT on the whole Switchboard+Fisher training set
steps/align_si.sh --nj 100 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   11500 400000 data/train_nodup data/lang exp/tri3a_ali exp/tri3b || exit 1;
(  utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph || exit 1;
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3b/graph data/eval2000 exp/tri3b/decode_dev || exit 1;
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3b/graph data/rt03 exp/tri3b/decode_rt03 || exit 1;
)&

# Next we'll use fMLLR and train with SAT (i.e. on 
# fMLLR features)

steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri3b exp/tri3b_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  11500 800000 data/train_nodup data/lang exp/tri3b_ali  exp/tri4a || exit 1;

( utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/eval2000 exp/tri4a/decode_dev || exit 1;
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/rt03 exp/tri4a/decode_rt03 || exit 1;
)&


steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4a exp/tri4a_ali || exit 1;


steps/train_sat.sh  --cmd "$train_cmd" \
  11500 1600000 data/train_nodup data/lang exp/tri4a_ali  exp/tri5a || exit 1;

( utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri5a/graph data/eval2000 exp/tri5a/decode_dev || exit 1;
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri5a/graph data/rt03 exp/tri5a/decode_rt03 || exit 1;
)&

steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri5a exp/tri5a_ali || exit 1;


steps/train_sat.sh  --cmd "$train_cmd" \
  11500 3200000 data/train_nodup data/lang exp/tri5a_ali  exp/tri6a || exit 1;

( utils/mkgraph.sh data/lang_test exp/tri6a exp/tri6a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri6a/graph data/eval2000 exp/tri6a/decode_dev || exit 1;
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri6a/graph data/rt03 exp/tri6a/decode_rt03 || exit 1;
)&

# Optional tri6a alignment for further training purposes

#steps/align_fmllr.sh --nj 200 --cmd "$train_cmd" \
#  data/train_nodup data/lang exp/tri6a exp/tri6a_ali || exit 1;

