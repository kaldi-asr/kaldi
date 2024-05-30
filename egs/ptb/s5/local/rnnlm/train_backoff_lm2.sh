#!/usr/bin/env bash

# this script is to train a small, pruned n-gram backoff LM to be used for sampling
# purposes during RNNLM training.  It uses the C++ tool that we wrote for this
# purpose, which creates an LM optimized specifically for this task.


dir=exp/rnnlm_data_prep

# To be run from the directory egs/ptb/s5.
# to be run after prepare_rnnlm_data.sh.  this will all later be refactored.

. ./path.sh
set -e


# When we start running this with multiple datasets we'll have to
# incorporate the data weights.  This is triial.



# old version that writes the ARPA file:
#rnnlm-get-sampling-lm --discounting-constant=1.0 \
#  --unigram-factor=200.0 --backoff-factor=2.0 \
#   "cat data/text/ptb.txt | utils/sym2int.pl data/vocab/words.txt | awk '{print 1.0, \$0}' |" \
#   data/vocab/words.txt "| gzip -c > $dir/lm.arpa.gz"



vocab_size=$(tail -n 1 data/vocab/words.txt |awk '{print $NF + 1}')

rnnlm-get-sampling-lm --discounting-constant=1.0 \
                      --unigram-factor=200.0 --backoff-factor=2.0 \
                      --vocab-size=$vocab_size \
   "cat data/text/ptb.txt | utils/sym2int.pl data/vocab/words.txt | awk '{print 1.0, \$0}' |" \
  $dir/sampling.lm


exit 0
############## BEYOND HERE IS COMMENT

# baseline speed test with pocolm-based LM:
time rnnlm-get-egs --vocab-size=10003 data/vocab/words.txt 'gunzip -c data/pocolm/trigram_100k.arpa.gz|' 'head -n 10000 exp/rnnlm_data_prep/text/1.txt|' ark:/dev/null
# 23 seconds


# and the size of the pocolm lm:
gunzip -c data/pocolm/trigram_100k.arpa.gz|head -n 4
\data\
ngram 1=10001
ngram 2=69769
ngram 3=23799



# the size of our LM:
gunzip -c exp/rnnlm_data_prep/lm.arpa.gz|head -n 4
\data\
ngram 1=10002
ngram 2=17697
ngram 3=14678

time rnnlm-get-egs --vocab-size=10003 data/vocab/words.txt 'gunzip -c exp/rnnlm_data_prep/lm.arpa.gz|' 'head -n 10000 exp/rnnlm_data_prep/text/1.txt|' ark:/dev/null
# 9.8 seconds.
