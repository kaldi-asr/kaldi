#!/bin/bash

# Copyright 2017 Tsinghua University (Author: Zhiyuan Tang, Dong Wang)
# Apache 2.0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh


## Download the data for AP17-OLR Challenge using local/olr/download_data.sh
## License agreement is required.
## Assume that the data is downloaded in data/

echo "---- make fbank features ----"
for i in data/{train,dev_1s,dev_3s,dev_all}; do
  steps/make_fbank.sh --nj 8 --cmd "$cpu_cmd" $i || exit 1;
  feat-to-len scp:$i/feats.scp ark,t:$i/feats.len || exit 1;
done

echo "---- generate language alignment for train set ----"
python local/olr/gen_ali.py data/train || exit 1;

## train TDNN for LID, using local/olr/run_tdnn_raw.sh
## or an LSTM for LID, using local/olr/run_lstm_raw.sh
## we prefer Phonetic Temporal Neural (PTN) LID (https://arxiv.org/abs/1705.03151)
## PTN performs better than both above, even better than i-vector LID with short utterances.
echo "---- train and evaluate PTN LID ----"
local/olr/run_ptn.sh || exit 1;

exit 0;

