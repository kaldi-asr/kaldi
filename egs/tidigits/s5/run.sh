#!/bin/bash

# Note: this TIDIGITS setup has not been tuned at all and has some obvious
# deficiencies; this has been created as a starting point for a tutorial.
# We're just using the "adults" data here, not the data from children.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

tidigits=/export/corpora5/LDC/LDC93S10
#tidigits=/mnt/matylda2/data/TIDIGITS

# The following command prepares the data/{train,dev,test} directories.
local/tidigits_data_prep.sh $tidigits || exit 1;
local/tidigits_prepare_lang.sh  || exit 1;
utils/validate_lang.pl data/lang/ # Note; this actually does report errors,
   # and exits with status 1, but we've checked them and seen that they
   # don't matter (this setup doesn't have any disambiguation symbols,
   # and the script doesn't like that).

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in test train; do
 steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

utils/subset_data_dir.sh data/train 1000 data/train_1k



# try --boost-silence 1.25 to some of the scripts below (also 1.5, if that helps...
# effect may not be clear till we test triphone system.  See
# wsj setup for examples (../../wsj/s5/run.sh)

steps/train_mono.sh  --nj 4 --cmd "$train_cmd" \
  data/train_1k data/lang exp/mono0a

 utils/mkgraph.sh data/lang exp/mono0a exp/mono0a/graph && \
 steps/decode.sh --nj 10 --cmd "$decode_cmd" \
      exp/mono0a/graph data/test exp/mono0a/decode

steps/align_si.sh --nj 4 --cmd "$train_cmd" \
   data/train data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --cmd "$train_cmd" \
    300 3000 data/train data/lang exp/mono0a_ali exp/tri1


 utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
 steps/decode.sh --nj 10 --cmd "$decode_cmd" \
      exp/tri1/graph data/test exp/tri1/decode

# Example of looking at the output.
# utils/int2sym.pl -f 2- data/lang/words.txt  exp/tri1/decode/scoring/19.tra | sed "s/ $//" | sort | diff - data/test/text


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep SER $x/wer_* | utils/best_wer.sh; done

#exp/mono0a/decode/wer_17:%SER 3.67 [ 319 / 8700 ]
#exp/tri1/decode/wer_19:%SER 2.64 [ 230 / 8700 ]
