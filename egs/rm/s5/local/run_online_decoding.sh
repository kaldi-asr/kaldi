#!/bin/bash

. cmd.sh


steps/online/prepare_online_decoding.sh --cmd "$train_cmd" data/train data/lang \
    exp/tri3b exp/tri3b_mmi/final.mdl exp/tri3b_online/


steps/online/decode.sh --cmd "$decode_cmd" --nj 10 exp/tri3b/graph \
  data/train exp/tri3b_online/test

