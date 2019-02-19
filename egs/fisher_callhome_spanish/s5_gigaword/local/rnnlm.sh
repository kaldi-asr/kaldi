#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains LMs on the swbd LM-training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 35) was 34, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 41.9 / 50.0.
# Train objf: -5.07 -4.43 -4.25 -4.17 -4.12 -4.07 -4.04 -4.01 -3.99 -3.98 -3.96 -3.94 -3.92 -3.90 -3.88 -3.87 -3.86 -3.85 -3.84 -3.83 -3.82 -3.81 -3.80 -3.79 -3.78 -3.78 -3.77 -3.77 -3.76 -3.75 -3.74 -3.73 -3.73 -3.72 -3.71
# Dev objf:   -10.32 -4.68 -4.43 -4.31 -4.24 -4.19 -4.15 -4.13 -4.10 -4.09 -4.05 -4.03 -4.02 -4.00 -3.99 -3.98 -3.98 -3.97 -3.96 -3.96 -3.95 -3.94 -3.94 -3.94 -3.93 -3.93 -3.93 -3.92 -3.92 -3.92 -3.92 -3.91 -3.91 -3.91 -3.91


dir=Spanish_gigawrd/rnnlm
pocolm_dir=Spanish_gigawrd/work_pocolm/lm/110000_3.pocolm_pruned
wordslist=
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=0
train_stage=-30
text=Spanish_gigawrd/text_lm
text_dir=Spanish_gigawrd/text_lm

. ./cmd.sh
. ./utils/parse_options.sh

mkdir -p $dir/config
set -e

for f in $text/dev.txt; do
    [ ! -f $f ] && \
	echo "$0: expected file $f to exist;" && exit 1
done

if [ $stage -le 0 ]; then
    if [ -f $text_dir/unigram_weights ] ; then
	mv $text_dir/unigram_weights $pocolm_dir/
    fi
    cp $wordslist $dir/config/words.txt
    n=`cat $dir/config/words.txt | wc -l`
    echo "<brk> $n" >> $dir/config/words.txt

    # words that are not present in words.txt but are in the training or dev data, will be
    # mapped to <SPOKEN_NOISE> during training.
    echo "<unk>" >$dir/config/oov.txt
    local/get_data_weights.pl $pocolm_dir $dir/config/data_weights.txt 
    rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
				 --unk-word="<unk>" \
				 --data-weights-file=$dir/config/data_weights.txt \
				 $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt
    
      # choose features
      rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
			       --use-constant-feature=true \
			       --special-words='<s>,</s>,<brk>,<unk>,[noise],[laughter]' \
			       $dir/config/words.txt > $dir/config/features.txt
fi

if [ $stage -le 1 ]; then
        cat <<EOF >$dir/config/xconfig 
	input dim=$embedding_dim name=input
	relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
	fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
	relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-3))
	fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
	relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-3))
	output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
	rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
    rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
    rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 2 \
			 --stage $train_stage --num-epochs 5 --cmd "$train_cmd" $dir
fi

exit 0
