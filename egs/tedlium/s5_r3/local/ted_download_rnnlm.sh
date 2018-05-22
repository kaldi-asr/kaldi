#!/bin/bash
#
# Copyright  2018  François Hernandez
# Apache 2.0
#
# This script downloads pre-built RNN language models trained on the TED-LIUM
# text data and acoustic training data.  If you want to build these
# models yourself, run the script local/ted_train_rnnlm.sh.

set -e

echo "$0: downloading Tedlium RNNLM models (it won't re-download if it was already downloaded.)"
wget --continue http://kaldi-asr.org/models/6/tedlium_rnnlm.tgz -P exp/rnnlm_lstm_tdnn_a_averaged || exit 1
tar -xvzf exp/rnnlm_lstm_tdnn_a_averaged/tedlium_rnnlm.tgz || exit 1
rm exp/rnnlm_lstm_tdnn_a_averaged/tedlium_rnnlm.tgz

exit 0