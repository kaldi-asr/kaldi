#!/usr/bin/env bash

# this script generates Pocolm-estimated language models with various
# data sources in data/text folder and places the output in data/lm.

set -euo pipefail

. ./path_venv.sh

export POCOLM_ROOT=$(cd $KALDI_ROOT/tools/pocolm/; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts


wordlist=None
num_word=100000
pocolm_stage=2
ngram_order=3
lm_dir=
arpa_dir=
textdir=
max_memory='--max-memory=8G'

. ./cmd.sh
. ./utils/parse_options.sh


# If you do not want to set memory limitation for "sort", you can use
#max_memory=
# Choices for the max-memory can be:
# 1) integer + 'K', 'M', 'G', ...
# 2) integer + 'b', meaning unit is byte and no multiplication
# 3) integer + '%', meaning a percentage of memory
# 4) integer, default unit is 'K'

fold_dev_opt=
# If you want to fold the dev-set in to the 'swbd1' set to produce the final
# model, un-comment the following line.  For use in the Kaldi example script for
# ASR, this isn't suitable because the 'dev' set is the first 10k lines of the
# switchboard data, which we also use as dev data for speech recognition
# purposes.
#fold_dev_opt="--fold-dev-into=swbd1"

bypass_metaparam_optim_opt=
# If you want to bypass the metaparameter optimization steps with specific metaparameters
# un-comment the following line, and change the numbers to some appropriate values.
# You can find the values from output log of train_lm.py.
# These example numbers of metaparameters is for 3-gram model running with train_lm.py.
# the dev perplexity should be close to the non-bypassed model.
#bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.091,0.867,0.753,0.275,0.100,0.018,0.902,0.371,0.183,0.070"
# Note: to use these example parameters, you may need to remove the .done files
# to make sure the make_lm_dir.py be called and tain only 3-gram model
#for order in 3; do
#rm -f ${lm_dir}/${num_word}_${order}.pocolm/.done

limit_unk_history_opt=
# If you want to limit the left of <unk> in the history of a n-gram
# un-comment the following line
#limit_unk_history_opt="--limit-unk-history=true"

for order in ${ngram_order}; do
  # decide on the vocabulary.
  # Note: you'd use --wordlist if you had a previously determined word-list
  # that you wanted to use.
  lm_name="${num_word}_${order}"
  min_counts=''
  # Note: the following might be a more reasonable setting:
  # min_counts='fisher=2 swbd1=1'
  if [ -n "${min_counts}" ]; then
    lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "=" "-"`"
  fi
  unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm
  train_lm.py --num-words=${num_word} --num-splits=5 --warm-start-ratio=10 ${max_memory} \
              --min-counts=${min_counts} \
              --keep-int-data=true ${fold_dev_opt} ${bypass_metaparam_optim_opt} \
              ${limit_unk_history_opt} ${textdir} ${order} ${lm_dir}/work ${unpruned_lm_dir}

  if [ $pocolm_stage -eq 2 ];then
  mkdir -p ${arpa_dir}
  format_arpa_lm.py ${max_memory} ${unpruned_lm_dir} | gzip -c > ${arpa_dir}/${lm_name}_${order}gram_unpruned.arpa.gz

  # example of pruning.  note: the threshold can be less than or more than one.
  get_data_prob.py ${max_memory} ${textdir}/dev.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
  for threshold in 1.0 2.0 4.0; do
    pruned_lm_dir=${lm_dir}/${lm_name}_prune${threshold}.pocolm
    prune_lm_dir.py --final-threshold=${threshold} ${max_memory} ${unpruned_lm_dir} ${pruned_lm_dir} 2>&1 | tail -n 5 | head -n 3
    get_data_prob.py ${max_memory} ${textdir}/dev.txt ${pruned_lm_dir} 2>&1 | grep -F '[perplexity'

    format_arpa_lm.py ${max_memory} ${pruned_lm_dir} | gzip -c > ${arpa_dir}/${lm_name}_${order}gram_prune${threshold}.arpa.gz

  done

  # example of pruning by size.
  size=1000000
  pruned_lm_dir=${lm_dir}/${lm_name}_prune${size}.pocolm
  prune_lm_dir.py --target-num-ngrams=${size} ${max_memory} ${unpruned_lm_dir} ${pruned_lm_dir} 2>&1 | tail -n 8 | head -n 6 | grep -v 'log-prob changes'
  get_data_prob.py ${textdir}/dev.txt ${max_memory} ${pruned_lm_dir} 2>&1 | grep -F '[perplexity'

  format_arpa_lm.py ${max_memory} ${pruned_lm_dir} | gzip -c > ${arpa_dir}/${lm_name}_${order}gram_prune${size}.arpa.gz
  fi
done

# (run local/srilm_baseline.sh ${num_word} to see the following result e.g. local/srilm_baseline.sh 40000 )

# the following does does some self-testing, including
# that the computed derivatives are accurate.
# local/self_test.sh

# perplexities from pocolm-estimated language models with pocolm's interpolation
# method from orders 3, 4, and 5 are:
# order 3: optimize_metaparameters.py: final perplexity without barrier function was -4.358818 (perplexity: 78.164689)
# order 4: optimize_metaparameters.py: final perplexity without barrier function was -4.309507 (perplexity: 74.403797)
# order 5: optimize_metaparameters.py: final perplexity without barrier function was -4.301741 (perplexity: 73.828181)

# note, the perplexities from pocolm-estimated language models with SRILM's
# interpolation from orders 3 and 4 are (from local/pocolm_with_srilm_combination.sh),
# 78.8449 and 75.2202 respectively.

# note, the perplexities from SRILM-estimated language models with SRILM's
# interpolation tool from orders 3 and 4 are (from local/srilm_baseline.sh),
# 78.9056 and 75.5528 respectively.
