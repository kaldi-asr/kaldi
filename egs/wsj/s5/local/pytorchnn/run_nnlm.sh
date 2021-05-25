#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# This script trains an RNN (LSTM and GRU) or Transformer-based
# language model with PyTorch and performs N-best and lattice rescoring.
# The N-best rescoring is in a batch computation mode as well.

# Dev/eval92 perplexities of the Transformer LM used for rescoring are: 55.7/71.1
# Baseline WER with a 4-gram LM:
# %WER 2.36 [ 133 / 5643, 10 ins, 11 del, 112 sub ] exp/chain/tdnn1g_sp/decode_bd_tgpr_eval92_fg//wer_13_1.0
# N-best rescoring:
# %WER 1.63 [ 92 / 5643, 7 ins, 5 del, 80 sub ] exp/chain/tdnn1g_sp/decode_bd_tgpr_eval92_fg_pytorch_transformer_nbest//wer_10_0.0
# Lattice rescoring:
# %WER 1.58 [ 89 / 5643, 6 ins, 6 del, 77 sub ] exp/chain/tdnn1g_sp/decode_bd_tgpr_eval92_fg_pytorch_transformer//wer_10_0.0

# Begin configuration section.
stage=0
ac_model_dir=exp/chain/tdnn1g_sp
decode_dir_suffix=pytorch_transformer
pytorch_path=exp/pytorch_transformer
nn_model=$pytorch_path/model.pt

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=768
hidden_dim=768
nlayers=8
nhead=8
learning_rate=0.1
seq_len=100
dropout=0.2

oov='<UNK>' # Symbol for out-of-vocabulary words

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e

text=data/local/dict_nosp_larger/cleaned.gz
test_data=data/test_eval92/text
wordlist=data/lang_nosp/words.txt
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
  # Text data preparation for training a neural LM.
  for f in $text $test_data $wordlist; do
    [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1
  done
  mkdir -p $data_dir
  echo -n >$data_dir/valid.txt
  # hold out one in every 500 lines as valid data.
  gunzip -c $text | awk -v data_dir=$data_dir '{if(NR%500 == 0) { print >data_dir"/valid.txt"; } else {print;}}' >$data_dir/train.txt
  < $test_data cut -d ' ' -f2- > $data_dir/test.txt
  cp $wordlist $data_dir/
  # Make sure words.txt contains the symbol for out-of-vocabulary words.
  if ! grep -w $oov $data_dir/words.txt >/dev/null; then
    n=$(wc -l < $data_dir/words.txt)
    echo "$oov $n" >> $data_dir/words.txt
  fi
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
            --oov "'$oov'" \
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

LM=bd_tgpr
if [ $stage -le 2 ]; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in eval92; do
      decode_dir=${ac_model_dir}/decode_${LM}_${decode_set}_fg
      steps/pytorchnn/lmrescore_nbest_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --N 20 \
        --weight 0.7 \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --oov-symbol "'$oov'" \
        data/lang_test_$LM $nn_model $data_dir/words.txt \
        data/test_${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

if [ $stage -le 3 ]; then
  echo "$0 Perform lattice rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in eval92; do
      decode_dir=${ac_model_dir}/decode_${LM}_${decode_set}_fg
      steps/pytorchnn/lmrescore_lattice_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.7 \
        --beam 4 \
        --epsilon 0.5 \
        --oov-symbol "'$oov'" \
        data/lang_test_$LM $nn_model $data_dir/words.txt \
        data/test_${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}
  done
fi

exit 0
