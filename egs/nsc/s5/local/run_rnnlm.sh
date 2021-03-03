#!/usr/bin/env bash

# Copyright 2014  Yandex (Author: Ilya Edrenkin)
# Apache 2.0

# Begin configuration section.
rnnlm_options="-hidden 150 -direct 1000 -direct-order 5"
rnnlm_tag="h150_me5-1000"
num_threads=8 # set this value to the number of physical cores on your CPU
stage=0
rnnlm_ver=faster-rnnlm
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./path.sh
. utils/parse_options.sh

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-dir> <lm-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/lm data/local/lm"
  echo ", where:"
  echo "    <data-dir> is the directory in which the text corpus is downloaded"
  echo "    <lm-dir> is the directory in which the language model is stored"
  echo "Main options:"
  echo "  --rnnlm-options <int>   # default '$rnnlm_options'. Command line arguments to pass to rnnlm"
  echo "  --rnnlm-tag <str>       # default '$rnnlm_tag' The tag is appended to exp/ folder name"
  echo "  --num-threads <int>     # default 16. Number of concurrent threadss to train RNNLM"
  echo "  --stage <int>           # 1 to download and prepare data, 2 to train RNNLM, 3 to rescore tri6b with a trained RNNLM"
  exit 1
fi

s5_dir=`pwd`
data_dir=`utils/make_absolute.sh $1`
lm_dir=`utils/make_absolute.sh $2`
modeldir=data/lang_${rnnlm_ver}_${rnnlm_tag}

if [ $stage -le 1 ]; then
  echo "$0: Prepare training data for RNNLM"
  cd $data_dir
  if [ -f "librispeech-lm-norm.dev.txt" ]; then
      echo "$0: SKIP File librispeech-lm-norm.dev.txt already exists"
  else
      wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
      gunzip librispeech-lm-norm.txt.gz
      $s5_dir/utils/filt.py $lm_dir/librispeech-vocab.txt librispeech-lm-norm.txt | shuf > librispeech-lm-norm.train.txt
      $s5_dir/utils/filt.py $lm_dir/librispeech-vocab.txt <(awk '{$1=""; print $0}' $s5_dir/data/train_960/text) > librispeech-lm-norm.dev.txt.tmp
      mv librispeech-lm-norm.dev.txt.tmp librispeech-lm-norm.dev.txt
      rm librispeech-lm-norm.txt
  fi
  cd $s5_dir

fi

if [ $stage -le 2 ]; then
  echo "$0: Training RNNLM. It will probably take several hours."
  $KALDI_ROOT/tools/extras/check_for_rnnlm.sh "$rnnlm_ver" || exit 1
  rnnlm_path="$(utils/make_absolute.sh $KALDI_ROOT)/tools/$rnnlm_ver/rnnlm"
  cd $s5_dir
  mkdir -p $modeldir
  echo "$0: Model file: $modeldir/rnnlm"
  if [ -f "$modeldir/rnnlm" ]; then
      echo "$0: SKIP file '$modeldir/rnnlm' already exists"
  else
      rm -f $modeldir/rnnlm.tmp
      rnnlm_cmd="$rnnlm_path"
      if type taskset >/dev/null 2>&1 ; then
          # HogWild works much faster if all threads are binded to the same phisical cpu
          rnnlm_cmd="taskset -c $(seq -s, 0 $(( $num_threads - 1 )) | sed 's/,$//') $rnnlm_cmd"
      fi
      $rnnlm_cmd -rnnlm $modeldir/rnnlm.tmp \
          -train $data_dir/librispeech-lm-norm.train.txt \
          -valid $data_dir/librispeech-lm-norm.dev.txt \
          -threads $num_threads $rnnlm_options -retry 1 -stop 1.0 2>&1 | tee $modeldir/rnnlm.log
      touch $modeldir/unk.probs
      awk '{print $1}' $modeldir/rnnlm.tmp > $modeldir/wordlist.rnn
      mv $modeldir/rnnlm.tmp $modeldir/rnnlm
      mv $modeldir/rnnlm.tmp.nnet $modeldir/rnnlm.nnet
  fi
fi

if [ $stage -le 3 ]; then
  echo "$0: Performing RNNLM rescoring on tri6b decoding results"
  for lm in tgsmall tgmed tglarge; do
    for devset in dev_clean dev_other; do
      sourcedir=exp/tri6b/decode_${lm}_${devset}
      if [ ! -d "$sourcedir" ]; then
          echo "$0: WARNING cannot find source dir '$sourcedir' to rescore"
          continue
      fi
      resultsdir=${sourcedir}_${rnnlm_ver}_${rnnlm_tag}
      rm -rf ${resultsdir}_L0.5
      steps/rnnlmrescore.sh --skip_scoring false --rnnlm_ver $rnnlm_ver --N 100 0.5 data/lang_test_$lm $modeldir data/$devset $sourcedir ${resultsdir}_L0.5
      for coef in 0.25 0.75; do
          rm -rf ${resultsdir}_L${coef}
          cp -r ${resultsdir}_L0.5 ${resultsdir}_L${coef}
          steps/rnnlmrescore.sh --skip_scoring false --rnnlm_ver $rnnlm_ver --N 100 --stage 7 $coef data/lang_test_$lm $modeldir data/$devset $sourcedir ${resultsdir}_L${coef}
      done
    done
  done
fi
