#!/bin/bash

# Copyright 2014  Yandex (Author: Ilya Edrenkin)
# Apache 2.0

# Begin configuration section.  
hidden=150
maxent_order=5
maxent_size=1000
num_threads=16
stage=0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. path.sh
. utils/parse_options.sh

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-dir> <lm-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/lm data/local/lm"
  echo ", where:"
  echo "    <data-dir> is the directory in which the text corpus is downloaded"
  echo "    <lm-dir> is the directory in which the language model is stored"
  echo "Main options:"
  echo "  --hidden <int>          # default 150. Hidden layer size"
  echo "  --maxent-order <int>    # default 5. Maxent features order size"
  echo "  --maxent-size <int>     # default 1000. Maxent features hash size"
  echo "  --num-threads <int>     # default 16. Number of concurrent threadss to train RNNLM"
  echo "  --stage <int>           # 1 to download and prepare data, 2 to train RNNLM, 3 to rescore tri6b with a trained RNNLM"
  exit 1
fi

s5_dir=`pwd`
data_dir=`readlink -f $1`
lm_dir=`readlink -f $2`
rnnlm_ver=rnnlm-hs-0.1b # Probably could make this an option, but Tomas's RNN will take long to train on 200K vocab
rnnlmdir=data/lang_rnnlm_h${hidden}_me${maxent_order}-${maxent_size}
export PATH=$KALDI_ROOT/tools/$rnnlm_ver:$PATH

if [ $stage -le 1 ]; then
  echo "$0: Prepare training data for RNNLM"
  cd $data_dir
  wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
  gunzip librispeech-lm-norm.txt.gz
  $s5_dir/utils/filt.py $lm_dir/librispeech-vocab.txt librispeech-lm-norm.txt | shuf > librispeech-lm-norm.train.txt
  $s5_dir/utils/filt.py $lm_dir/librispeech-vocab.txt <(awk '{$1=""; print $0}' $s5_dir/data/train_960/text) > librispeech-lm-norm.dev.txt
  rm librispeech-lm-norm.txt
  cd $s5_dir
  
fi

if [ $stage -le 2 ]; then
  echo "$0: Training RNNLM. It will probably take several hours."
  cd $KALDI_ROOT/tools
  if [ -f $rnnlm_ver/rnnlm ]; then
      echo "Not installing the rnnlm toolkit since it is already there."
  else
      extras/install_rnnlm_hs.sh
  fi
  cd $s5_dir
  mkdir -p $rnnlmdir
  rnnlm -rnnlm $rnnlmdir/rnnlm -train $data_dir/librispeech-lm-norm.train.txt -valid $data_dir/librispeech-lm-norm.dev.txt \
      -threads $num_threads -hidden $hidden -direct-order $maxent_order -direct $maxent_size -retry 1 -stop 1.0
  touch $rnnlmdir/unk.probs
  awk '{print $1}' $rnnlmdir/rnnlm > $rnnlmdir/wordlist.rnn
fi

if [ $stage -le 3 ]; then
  echo "$0: Performing RNNLM rescoring on tri6b decoding results"
  for lm in tgsmall tgmed; do
    for devset in dev_clean dev_other; do
      sourcedir=exp/tri6b/decode_pp_${lm}_${devset}
      resultsdir=${sourcedir}_rnnlm_h${hidden}_me${maxent_order}-${maxent_size}
      steps/rnnlmrescore.sh --rnnlm_ver $rnnlm_ver --N 100 0.5 data/lang_pp_test_$lm $rnnlmdir data/$devset $sourcedir ${resultsdir}_L0.5
      cp -r ${resultsdir}_L0.5 ${resultsdir}_L0.25
      cp -r ${resultsdir}_L0.5 ${resultsdir}_L0.75
      steps/rnnlmrescore.sh --rnnlm_ver $rnnlm_ver --N 100 --stage 7 0.25 data/lang_pp_test_$lm $rnnlmdir data/$devset $sourcedir ${resultsdir}_L0.25
      steps/rnnlmrescore.sh --rnnlm_ver $rnnlm_ver --N 100 --stage 7 0.75 data/lang_pp_test_$lm $rnnlmdir data/$devset $sourcedir ${resultsdir}_L0.75
    done
  done
fi


