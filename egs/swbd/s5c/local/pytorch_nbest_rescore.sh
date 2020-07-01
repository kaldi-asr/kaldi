#!/bin/bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2020  Ke Li

# This script performs N-best rescoring with a pretrained neural LM (RNNLM or Transformer) by PyTorch

# Dev/eval2000 perplexity of a 2-layer LSTM model is: 47.0/41.7. WERs by N-best rescoring (with hidden states carried over sentences) are as below:
# %WER 10.9 | 4459 42989 | 90.4 6.3 3.3 1.4 10.9 42.6 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorchrnnlm_nbest//score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 7.1 | 1831 21395 | 93.8 4.2 2.0 0.9 7.1 36.6 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorchrnnlm_nbest//score_11_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.7 | 2628 21594 | 87.2 8.6 4.2 2.0 14.7 46.8 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorchrnnlm_nbest//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys
# Without hidden state carry-over, the WER on eval2000 by N-best rescoring with the LSTM model is 11.2.

# Dev/eval2000 perplexity of a Transformer LM is: 46.8/41.5. WERs by N-best rescoring are as below:
# %WER 10.9 | 4459 42989 | 90.5 6.3 3.1 1.5 10.9 42.2 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorchtransformer_nbest//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 7.1 | 1831 21395 | 93.8 4.1 2.1 0.9 7.1 36.4 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorchtransformer_nbest//score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.7 | 2628 21594 | 87.3 8.5 4.2 2.0 14.7 46.2 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorchtransformer_nbest//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys

# Note: both the LSTM and Transformer models are trained on the same dataset as Kaldi RNNLM does.

# Begin configuration section.
stage=-10
ac_model_dir=exp/chain/tdnn7q_sp
decode_dir_suffix=pytorchrnnlm
# Users need to provide the path to a pretrained PyTorch model and the corresponding
# vocab (assume the vocab is in a sub directory data/)
pytorch_path=
nn_model=$pytorch_path/pretrained_model
vocabulary=$pytorch_path/data/words.txt
model_type='RNNLM' # RNNLM or Transformer

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

for f in $nn_model $vocabulary; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist." && exit 1
done

if python steps/pytorchnn/check_py.py 2>/dev/null; then
  echo PyTorch is ready to use on the python side. This is good.
else
  echo PyTorch not found on the python side.
  echo Please install PyTorch first. For example, you can install it with conda:
  echo "conda install pytorch torchvision cudatoolkit=10.1 -c pytorch", or
  echo with pip: "pip install torch torchvision". If you already have PyTorch
  echo installed somewhere else, you need to add it to your PATH.
  exit 1
fi

LM=sw1_fsh_fg # using the 4-gram const arpa file as old lm
if [ $stage -le 0 ]; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir by a PyTorch trained neural LM."
  for decode_set in eval2000; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      rnnlm/lmrescore_nbest_pytorchmodel.sh \
        --cmd "$decode_cmd --mem 4G"  --N 20 \
        0.8 data/lang_$LM $nn_model $model_type $vocabulary \
        data/${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_nbest
  done
  wait
fi
exit 0
