#!/bin/bash

# This is neural net training on top of adapted 40-dimensional features.
# This version of the script uses GPUs.  We distinguish it by putting "_gpu"
# at the end of the directory name.
#
# Since we're using one quarter the number of jobs (num-jobs-nnet) as the
# run_4c.sh script, we halve the learning rate (generally speaking, splitting
# the difference like this is probably a good idea.)


. cmd.sh

# We increase the beam relative to the defaults; this is just for this RM setup,
# where the default beams will likely generate very thin lattices.  Note: the
# transform-dir is important to specify, since this system is on top of fMLLR
# features.

nj=8

steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
      --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
      --beam 20.0 --lattice-beam 10.0 \
      --transform-dir exp/tri3b_ali \
     data/train data/lang exp/nnet4c exp/nnet4c_denlats

steps/nnet2/align.sh  --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
      --transform-dir exp/tri3b_ali \
      --nj $nj data/train data/lang exp/nnet4c exp/nnet4c_ali

steps/nnet2/train_discriminative.sh --cmd "$decode_cmd" \
    --num-jobs-nnet 2 data/train data/lang \
    exp/nnet4c_ali exp/nnet4c_denlats exp/nnet4c/final.mdl exp/nnet5c_mpe

for epoch in 1 2 3 4; do
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch$epoch \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/nnet5c_mpe/decode_epoch$epoch  &

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch$epoch \
     --transform-dir exp/tri3b/decode_ug \
     exp/tri3b/graph_ug data/test exp/nnet5c_mpe/decode_ug_epoch$epoch &
done


