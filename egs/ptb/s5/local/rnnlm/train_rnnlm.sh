#!/usr/bin/bash


# this will eventually be totally refactored and moved into steps/.

dir=exp/rnnlm_data_prep
vocab=data/vocab/words.txt
embedding_dim=200
# work out the number of splits.
ns=$(rnnlm/get_num_splits.sh 200000 data/text $dir/data_weights.txt)

# split the data into pieces that individual jobs will train on.
# rnnlm/split_data.sh data/text $ns


rnnlm/prepare_split_data.py --vocab-file=$vocab --data-weights-file=$dir/data_weights.txt \
                            --num-splits=$ns data/text  $dir/text

. ./path.sh

# make config file.
# for now it's not even recurrent, it's just a feedforward network.

embedding_dim=200
cat >$dir/config <<EOF
input-node name=input dim=$embedding_dim
component name=affine1 type=NaturalGradientAffineComponent input-dim=$embedding_dim output-dim=$embedding_dim
component-node input=input name=affine1 component=affine1
output-node input=affine1 name=output
EOF



vocab_size=$(tail -n 1 $vocab |awk '{print $NF + 1}')

rnnlm/initialize_matrix.py --num-rows=$vocab_size --num-cols=$embedding_dim > $dir/embedding.0.mat

nnet3-init $dir/config $dir/0.rnnlm



utils/sym2int.pl  -f 2-  $vocab <data/text/split5/1.txt | utils/apply_map.pl -f 1 $dir/data_weights.txt | head

data/text/split5/1.txt

data/text/split5


/export/a09/dpovey/kaldi-chain/src/rnnlmbin/


rnnlm-train --use-gpu=no --read-rnnlm=$dir/0.rnnlm --write-rnnlm=/dev/null --read-embedding=$dir/embedding.0.mat \
  'ark:rnnlm-get-egs --vocab-size=$vocab_size $dir/text/1.txt ark,t:- |'
