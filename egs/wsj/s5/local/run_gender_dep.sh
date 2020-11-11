#!/usr/bin/env bash

# This script is not really finished, all it does is train a model with its
# means adapted to the female data, to demonstrate MAP adaptation.  To have real
# gender dependent decoding (which anyway we're not very enthused about), we
# would have to train both models, do some kind of gender identification, and
# then decode.  Or we could use the gender information in the test set.  But
# anyway that's not a direction we really want to go right now.

. ./cmd.sh

awk '{if ($2 == "f") { print $1; }}' < data/train_si84/spk2gender > spklist

utils/subset_data_dir.sh --spk-list spklist data/train_si84 data/train_si84_f

steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  data/train_si84_f data/lang exp/tri2b exp/tri2b_ali_si84_f

steps/train_map.sh --cmd "$train_cmd" data/train_si84_f data/lang exp/tri2b_ali_si84_f exp/tri2b_f
