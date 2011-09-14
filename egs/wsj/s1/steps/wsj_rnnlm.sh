#!/bin/bash

. path.sh || exit 1;

echo "Training a small RNN model (should take 1 hour)"
time steps/train_rnnlm.sh 5 140 small.rnn

echo "Converting into KALDI format"
scripts/convert_rnnlm.sh small.rnn small.rnn.voc small.rnn.kaldi

echo "Rescoring (see rescore_rnnlm_example)!!"

echo "Adapting model (TODO)"
