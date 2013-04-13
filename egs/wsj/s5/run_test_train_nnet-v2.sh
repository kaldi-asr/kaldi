#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


false && \
{
  (mkdir data-fbank/test_dev93; cp data/test_dev93/* data-fbank/test_dev93; rm data-fbank/test_dev93/feats.scp)
  (mkdir data-fbank/test_eval92; cp data/test_eval92/* data-fbank/test_eval92; rm data-fbank/test_eval92/feats.scp)
  (mkdir data-fbank/train_si84; cp data/train_si84/* data-fbank/train_si84; rm data-fbank/train_si84/feats.scp)
  
  steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data-fbank/test_dev93 data-fbank/test_dev93/_log data-fbank/test_dev93/_data || exit 1
  steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data-fbank/test_eval92 data-fbank/test_eval92/_log data-fbank/test_eval92/_data || exit 1
  steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data-fbank/train_si84 data-fbank/train_si84/_log data-fbank/train_si84/_data || exit 1

  steps/compute_cmvn_stats.sh data-fbank/test_dev93 data-fbank/test_dev93/_log data-fbank/test_dev93/_data || exit 1
  steps/compute_cmvn_stats.sh data-fbank/test_eval92 data-fbank/test_eval92/_log data-fbank/test_eval92/_data || exit 1
  steps/compute_cmvn_stats.sh data-fbank/train_si84 data-fbank/train_si84/_log data-fbank/train_si84/_data || exit 1
}


cuda_cmd="queue.pl -q long.q@pcspeech-gpu"

# HYBRID SYSTEM
model_size=1500000
hid_layers=2
learn_rate=0.008
#false && \
{ # Train the MLP
  dir=exp/tri2a_nnet4L_TEST_TRAINING-v2
  ali=exp/tri2a_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet-v2.sh --model-size $model_size --hid-layers $hid_layers --learn-rate $learn_rate --apply-cmvn true \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # Decode
  steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --acwt 0.09 \
    exp/tri2a/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 && \
  steps/decode_nnet.sh --nj 8 --cmd "$decode_cmd" --acwt 0.09 \
    exp/tri2a/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92
}



