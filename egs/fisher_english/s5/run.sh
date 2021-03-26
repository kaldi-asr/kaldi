#!/usr/bin/env bash

# It's best to run the commands in this one by one.

. ./cmd.sh
. ./path.sh
mfccdir=`pwd`/mfcc
set -e

# the next command produces the data in local/train_all
local/fisher_data_prep.sh /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
   /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13
# You could also try specifying the --calldata argument to this command as below.
# If specified, the script will use actual speaker personal identification 
# numbers released with the dataset, i.e. real speaker IDs. Note: --calldata has
# to be the first argument of this script.
# local/fisher_data_prep.sh --calldata /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
#    /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13

# at BUT:
# local/fisher_data_prep.sh /mnt/matylda6/jhu09/qpovey/FISHER/LDC2005T19 /mnt/matylda2/data/FISHER/

local/fisher_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

local/fisher_train_lms.sh 
local/fisher_create_test_lang.sh

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.

utils/fix_data_dir.sh data/train_all

steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train_all exp/make_mfcc/train_all $mfccdir || exit 1;

utils/fix_data_dir.sh data/train_all
utils/validate_data_dir.sh data/train_all


# The dev and test sets are each about 3.3 hours long.  These are not carefully
# done; there may be some speaker overlap with each other and with the training
# set.  Note: in our LM-training setup we excluded the first 10k utterances (they
# were used for tuning but not for training), so the LM was not (directly) trained
# on either the dev or test sets.
utils/subset_data_dir.sh --first data/train_all 10000 data/dev_and_test
utils/subset_data_dir.sh --first data/dev_and_test 5000 data/dev
utils/subset_data_dir.sh --last data/dev_and_test 5000 data/test
rm -r data/dev_and_test

steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev $mfccdir 
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir 

n=$[`cat data/train_all/segments | wc -l` - 10000]
utils/subset_data_dir.sh --last data/train_all $n data/train
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir 


# Now-- there are 1.6 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.

utils/subset_data_dir.sh --shortest data/train 100000 data/train_100kshort
utils/subset_data_dir.sh  data/train_100kshort 10000 data/train_10k
utils/data/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup
utils/subset_data_dir.sh --speakers data/train 30000 data/train_30k
utils/subset_data_dir.sh --speakers data/train 100000 data/train_100k


# The next commands are not necessary for the scripts to run, but increase 
# efficiency of data access by putting the mfcc's of the subset 
# in a contiguous place in a file.
( . ./path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_10k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_fish_10k_nodup.ark,$mfccdir/kaldi_fish_10k_nodup.scp \
  && cp $mfccdir/kaldi_fish_10k_nodup.scp data/train_10k_nodup/feats.scp
)
( . ./path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_30k/feats.scp{,.bak} 
  copy-feats scp:data/train_30k/feats.scp  ark,scp:$mfccdir/kaldi_fish_30k.ark,$mfccdir/kaldi_fish_30k.scp \
  && cp $mfccdir/kaldi_fish_30k.scp data/train_30k/feats.scp
)
( . ./path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_100k/feats.scp{,.bak} 
  copy-feats scp:data/train_100k/feats.scp  ark,scp:$mfccdir/kaldi_fish_100k.ark,$mfccdir/kaldi_fish_100k.scp \
  && cp $mfccdir/kaldi_fish_100k.scp data/train_100k/feats.scp
)

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a 

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k data/lang exp/mono0a_ali exp/tri1 || exit 1;


(utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1/graph data/dev exp/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/dev exp/tri2/decode_dev || exit 1;
)&


steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_100k data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/dev exp/tri3a/decode_dev || exit 1;
)&


# Next we'll use fMLLR and train with SAT (i.e. on 
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  5000 100000 data/train_100k data/lang exp/tri3a_ali  exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/dev exp/tri4a/decode_dev
)&


steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri4a exp/tri4a_ali || exit 1;


steps/train_sat.sh  --cmd "$train_cmd" \
  10000 300000 data/train data/lang exp/tri4a_ali  exp/tri5a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    exp/tri5a/graph data/dev exp/tri5a/decode_dev
)&

# this will help find issues with the lexicon.
# steps/cleanup/debug_lexicon.sh --nj 300 --cmd "$train_cmd" data/train_100k data/lang exp/tri5a data/local/dict/lexicon.txt exp/debug_lexicon_100k


# The step below won't run by default; it demonstrates a data-cleaning method.
# It doesn't seem to help in this setup; maybe the data was clean enough already.
# local/run_data_cleaning.sh

## The following is the best current neural net recipe.
# local/online/run_nnet2_multisplice.sh

# ## The following is an older recipe without multi-splice.
# # local/online/run_nnet2.sh


# ## The following is another older nnet recipe, on top of fMLLR features.
# # local/run_nnet2.sh
#

# This prepares lang directory with UNK modeled by a phone LM
# local/run_unk_model.sh

# These are semi-supervised training recipes using 50 hrs and 100 hrs 
# of supervised data respectively with 250 hrs of unsupervised data.
# run_50k.sh uses i-vector extractor trained on 300 hrs of combined data, 
# while run_100.sh uses i-vector extractor trained on 100 hrs of supervised data.
# run_50k.sh uses 4-gram LM trained on 1250 hrs transcripts, 
# while run_100k.sh uses 3-gram LM trained on 100 hrs transcripts.

# local/fisher_train_lms_pocolm.sh 
# local/fisher_create_test_lang.sh --arpa-lm data/local/pocolm/data/arpa/4gram_small.arpa.gz --dir data/lang_test_poco
# utils/build_const_arpa_lm.sh data/local/pocolm/data/arpa/4gram_big.arpa.gz data/lang_test_poco data/lang_test_poco_big

# for lang_dir in data/lang_test_poco; do
#   rm -r ${lang_dir}_unk ${lang_dir}_unk_big 2>/dev/null || true
#   cp -rT data/lang_unk ${lang_dir}_unk
#   cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst
#   cp -rT data/lang_unk ${lang_dir}_unk_big
#   cp ${lang_dir}_big/G.carpa ${lang_dir}_unk_big/G.carpa; 
# done

# Create supervised and unsupervised data subsets
# utils/subset_data_dir.sh --speakers data/train 100000 data/train_sup
# utils/subset_data_dir.sh --spk-list <(utils/filter_scp.pl --exclude data/train_sup/spk2utt data/train/spk2utt) data/train data/train_unsup100k
# utils/subset_data_dir.sh --speakers data/train_unsup100k 250000 data/train_unsup100k_250k

# local/semisup/run_50k.sh
# local/semisup/run_100k.sh 
