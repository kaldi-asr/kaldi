#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)



# Now make FBANK features.
# fbankdir should be some place with a largish disk where you
# want to store FBANK features.
fbankdir=fbank
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
 steps/make_fbank.sh --cmd "$train_cmd" --nj 20 \
   data-fbank/$x data/$x exp/make_fbank/$x $fbankdir || exit 1;
 steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done
# Make the SI-84 subset of FBANKs
utils/subset_data_dir.sh --first data-fbank/train_si284 7138 data-fbank/train_si84 || exit 1
steps/compute_cmvn_stats.sh data-fbank/train_si84 exp/make_fbank/train_si84 $fbankdir || exit 1;

# Align tri2a system with si84 data.
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train_si84 data/lang exp/tri2a exp/tri2a_ali_si84  || exit 1;
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  data/test_dev93 data/lang exp/tri2a exp/tri2a_ali_dev93  || exit 1;



model_size=3000000
hid_layers=2
learn_rate=0.008
( # Train the MLP
  dir=exp/tri2a_nnet4L
  ali=exp/tri2a_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --model-size $model_size --hid-layers $hid_layers --learn-rate $learn_rate --bunch-size 256 \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # Decode
  $mkgraph_cmd $dir/_mkgraph.log utils/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --acwt 0.09 \
    $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 && \
  steps/decode_nnet.sh --nj 8 --cmd "$decode_cmd" --acwt 0.09 \
    $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92
)


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
