#!/bin/bash


# This script shows how you can do data-cleaning, and exclude data that has a
# higher likelihood of being wrongly transcribed.  see the RESULTS file; this
# made essentially no difference in our case-- indicating, perhaps, that Fisher
# transcripts are already clean enough.


. cmd.sh
. path.sh
set -e


steps/cleanup/find_bad_utts.sh --nj 200 --cmd "$train_cmd" data/train data/lang \
  exp/tri5a exp/tri5a_cleanup

 # with threshold of 0.05 we keep 1.1 million out of 1.6 million utterances, and
 # around 8.7 million out of 18.1 million words
 # with threshold of 0.1 we keep 1.3 out of 1.6 million utterances, and around
 # 13.2 million out of 18.1 million words.
thresh=0.1
cat exp/tri5a_cleanup/all_info.txt | awk -v threshold=$thresh '{ errs=$2;ref=$3; if (errs <= threshold*ref) { print $1; } }' > uttlist
utils/subset_data_dir.sh --utt-list uttlist data/train data/train.thresh$thresh

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train.thresh$thresh data/lang exp/tri4a exp/tri4a_ali_$thresh

steps/train_sat.sh  --cmd "$train_cmd" \
  10000 300000 data/train data/lang exp/tri4a_ali_$thresh  exp/tri5a_$thresh || exit 1;


utils/mkgraph.sh data/lang_test exp/tri5a_$thresh exp/tri5a_$thresh/graph
steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
  exp/tri5a_$thresh/graph data/dev exp/tri5a_$thresh/decode_dev

