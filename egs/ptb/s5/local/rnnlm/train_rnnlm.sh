#!/usr/bin/bash


# this will eventually be totally refactored and moved into steps/.

dir=exp/rnnlm_data_prep
vocab=data/vocab/words.txt
embedding_dim=200
# work out the number of splits.
ns=$(rnnlm/get_num_splits.sh 200000 data/text $dir/data_weights.txt)
vocab_size=$(tail -n 1 $vocab |awk '{print $NF + 1}')

# split the data into pieces that individual jobs will train on.
# rnnlm/split_data.sh data/text $ns


rnnlm/prepare_split_data.py --vocab-file=$vocab --data-weights-file=$dir/data_weights.txt \
                            --num-splits=$ns data/text  $dir/text

. ./path.sh

# make config file.
# for now it's not even recurrent, it's just a feedforward network.

embedding_dim=600
cat >$dir/config <<EOF
input-node name=input dim=$embedding_dim
component name=affine1 type=NaturalGradientAffineComponent input-dim=$embedding_dim output-dim=$embedding_dim
component-node input=input name=affine1 component=affine1
output-node input=affine1 name=output
EOF



# note: this is way too slow, we need to speed it up somehow.
# I'm not sure if I want to have a dependency on numpy just for this though.
# maybe we can rewrite in perl.
rnnlm/initialize_matrix.py --num-rows=$vocab_size --num-cols=$embedding_dim \
                           --first-column=1.0 > $dir/embedding.0.mat

nnet3-init $dir/config $dir/0.rnnlm


rnnlm-train --use-gpu=no --read-rnnlm=$dir/0.rnnlm --write-rnnlm=$dir/1.rnnlm --read-embedding=$dir/embedding.0.mat \
  --write-embedding=/$dir/embedding.1.mat "ark:rnnlm-get-egs --vocab-size=$vocab_size $dir/text/1.txt ark,t:- |"


# just a note on the unigram entropy of PTB:
# awk '{for (n=1;n<=NF;n++) { count[$n]++; } count["</s>"]++; } END{ tot_count=0; tot_entropy=0.0; for(k in count) tot_count += count[k];  for (k in count) { p = count[k]*1.0/tot_count; tot_entropy += p*log(p); }  print "entropy is " -tot_entropy; }' <data/text/ptb.txt
# 6.52933
