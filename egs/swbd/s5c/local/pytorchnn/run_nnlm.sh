#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# This script trains an RNN (LSTM and GRU) or Transformer-based
# language model with PyTorch and performs N-best and lattice rescoring.
# More details about the lattice rescoring can be found in the paper:
# https://www.danielpovey.com/files/2021_icassp_parallel_lattice_rescoring.pdf
# The N-best rescoring is in a batch computation mode as well. It is thus much
# faster than N-best rescoring in local/rnnlm/run_tdnn_lstm.sh.

# Baseline WERs with a 4-gram LM:
# %WER 12.8 | 4459 42989 | 88.8 7.7 3.5 1.6 12.8 46.7 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.6 | 1831 21395 | 92.4 5.2 2.4 1.0 8.6 40.6 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg//score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 17.0 | 2628 21594 | 85.1 10.2 4.6 2.2 17.0 50.9 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys

# LSTM LM: Dev/eval2000 perplexities of a 2-layer LSTM model are: 47.1/41.9.
# N-best rescoring (with hidden states carried over sentences):
# %WER 10.9 | 4459 42989 | 90.5 6.4 3.1 1.4 10.9 42.7 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_nbest//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 7.1 | 1831 21395 | 93.8 4.1 2.1 0.9 7.1 36.4 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_nbest//score_11_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.6 | 2628 21594 | 87.3 8.5 4.1 1.9 14.6 46.7 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_nbest//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys
# Note: Without hidden-state-carry-over, the WER on eval2000 from N-best rescoring with the LSTM model is 11.2.
# Lattice rescoring (with hidden states carried over sentences):
# %WER 10.7 | 4459 42989 | 90.6 6.0 3.3 1.4 10.7 43.0 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm//score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 6.9 | 1831 21395 | 94.0 3.9 2.1 0.9 6.9 36.9 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm//score_11_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.5 | 2628 21594 | 87.3 8.1 4.6 1.8 14.5 47.2 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm//score_11_0.0/eval2000_hires.ctm.callhm.filt.sys

# Transformer LM: Dev/eval2000 perplexities of a Transformer LM are: 47.0/41.6.
# N-best rescoring:
# %WER 10.8 | 4459 42989 | 90.6 6.3 3.1 1.5 10.8 42.1 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer_nbest//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 7.2 | 1831 21395 | 93.7 4.2 2.1 1.0 7.2 37.3 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer_nbest//score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.4 | 2628 21594 | 87.6 8.3 4.1 2.0 14.4 45.5 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer_nbest//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys
# Lattice rescoring:
# %WER 10.6 | 4459 42989 | 90.8 6.0 3.1 1.5 10.6 42.0 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 6.8 | 1831 21395 | 94.1 3.8 2.1 0.9 6.8 36.2 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer//score_10_1.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.4 | 2628 21594 | 87.6 8.2 4.2 2.0 14.4 45.9 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys

# Begin configuration section.
stage=0
ac_model_dir=exp/chain/tdnn7q_sp
decode_dir_suffix=pytorch_transformer
pytorch_path=exp/pytorch_transformer
nn_model=$pytorch_path/model.pt

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 650 for LSTM (for reproducing perplexities and WERs above)
hidden_dim=512 # 650 for LSTM
nlayers=6 # 2 for LSTM
nhead=8
learning_rate=0.1 # 5 for LSTM
seq_len=100
dropout=0.1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e

data_dir=data/pytorchnn

# Check if PyTorch is installed to use with python
if python3 steps/pytorchnn/check_py.py 2>/dev/null; then
  echo PyTorch is ready to use on the python side. This is good.
else
  echo PyTorch not found on the python side.
  echo Please install PyTorch first. For example, you can install it with conda:
  echo "conda install pytorch torchvision cudatoolkit=10.2 -c pytorch", or
  echo with pip: "pip install torch torchvision". If you already have PyTorch
  echo installed somewhere else, you need to add it to your PATH.
  echo Note: you need to install higher version than PyTorch 1.1 to train Transformer models
  exit 1
fi

if [ $stage -le 0 ]; then
  local/pytorchnn/data_prep.sh $data_dir
fi

if [ $stage -le 1 ]; then
  # Train a PyTorch neural network language model.
  echo "Start neural network language model training."
  $cuda_cmd $pytorch_path/log/train.log utils/parallel/limit_num_gpus.sh \
    python3 steps/pytorchnn/train.py --data $data_dir \
            --model $model_type \
            --emsize $embedding_dim \
            --nhid $hidden_dim \
            --nlayers $nlayers \
            --nhead $nhead \
            --lr $learning_rate \
            --dropout $dropout \
            --seq_len $seq_len \
            --clip 1.0 \
            --batch-size 32 \
            --epoch 64 \
            --save $nn_model \
            --tied \
            --cuda
fi

LM=sw1_fsh_fg # Using the 4-gram const arpa file as old lm
if [ $stage -le 2 ]; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in eval2000; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      steps/pytorchnn/lmrescore_nbest_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --N 20 \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.8 \
        data/lang_$LM $nn_model $data_dir/words.txt \
        data/${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: Perform lattice rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in eval2000; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      steps/pytorchnn/lmrescore_lattice_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.8 \
        --beam 5 \
        --epsilon 0.5 \
        data/lang_$LM $nn_model $data_dir/words.txt \
        data/${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}
  done
fi

exit 0
