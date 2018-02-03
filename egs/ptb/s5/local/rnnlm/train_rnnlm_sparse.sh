#!/usr/bin/bash

# DEPRECATED.  See local/rnnlm/run_tdnn.sh.

# version that makes use of sparse features.
# this will eventually be totally refactored and moved into steps/.

dir=exp/rnnlm_data_prep
vocab=data/vocab/words.txt
embedding_dim=600
feat_dim=$(tail -n 1 $dir/features.txt | awk '{print $1 + 1;}')
# work out the number of splits.
ns=$(rnnlm/get_num_splits.sh 1000000 data/text $dir/data_weights.txt)
vocab_size=$(tail -n 1 $vocab |awk '{print $NF + 1}')

# split the data into pieces that individual jobs will train on.
# rnnlm/split_data.sh data/text $ns


rnnlm/prepare_split_data.py --vocab-file=$vocab --data-weights-file=$dir/data_weights.txt \
                            --num-splits=$ns data/text  $dir/text

. ./path.sh

# cat >$dir/config <<EOF
# input-node name=input dim=$embedding_dim
# component name=affine1 type=NaturalGradientAffineComponent input-dim=$embedding_dim output-dim=$embedding_dim
# component-node input=input name=affine1 component=affine1
# output-node input=affine1 name=output
# EOF

mkdir -p $dir/configs
cat >$dir/configs/network.xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=512 input=Append(0, IfDefined(-1))
relu-renorm-layer name=tdnn2 dim=512 input=Append(0, IfDefined(-2))
relu-renorm-layer name=tdnn3 dim=512 input=Append(0, IfDefined(-2))
output-layer name=output include-log-softmax=false dim=$embedding_dim learning-rate-factor=0.125 max-change=0.125
EOF

steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
nnet3-init $dir/configs/final.config - | nnet3-copy --learning-rate=0.0001 - $dir/0.rnnlm

# we may later initialize this to identity for the first block.
rnnlm/initialize_matrix.pl --first-element 1.0 --stddev 0.001 $feat_dim $embedding_dim > $dir/embedding.0.mat

# alternative path:
rnnlm/initialize_matrix.pl --first-column 1.0 $vocab_size $embedding_dim > $dir/word_embedding.0.mat


rnnlm-train --use-gpu=no --read-rnnlm=$dir/0.rnnlm --write-rnnlm=$dir/1.rnnlm --read-embedding=$dir/embedding.0.mat \
            --read-sparse-word-features=$dir/word_feats.txt \
            --write-embedding=/$dir/embedding.1.mat "ark:rnnlm-get-egs --vocab-size=$vocab_size $dir/text/1.txt ark,t:- |"

# or with GPU:
rnnlm-train --rnnlm.max-param-change=0.5 --embedding.max-param-change=0.5 \
            --read-sparse-word-features=$dir/word_feats.txt \
             --use-gpu=yes --read-rnnlm=$dir/0.rnnlm --write-rnnlm=$dir/1.rnnlm --read-embedding=$dir/embedding.0.mat \
             --write-embedding=$dir/embedding.1.mat 'ark:for n in 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1; do cat exp/rnnlm_data_prep/text/*.txt; done | rnnlm-get-egs --vocab-size=10003 - ark:- |'

# and evaluate on dev ata:
rnnlm-get-word-embedding $dir/word_feats.txt $dir/embedding.1.mat $dir/word_embedding.1.mat

# get the dev-data proability
rnnlm-get-egs --vocab-size=10003 $dir/text/dev.txt ark:- | \
 rnnlm-compute-prob --use-gpu=yes $dir/1.rnnlm $dir/word_embedding.1.mat ark:-

# with GPU, no sparse features.
rnnlm-train --rnnlm.max-param-change=0.5 --embedding.max-param-change=0.5 \
             --use-gpu=yes --read-rnnlm=$dir/0.rnnlm --write-rnnlm=$dir/1.rnnlm --read-embedding=$dir/word_embedding.0.mat \
            --write-embedding=$dir/embedding.1.mat 'ark:for n in 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1; do cat exp/rnnlm_data_prep/text/*.txt; done | rnnlm-get-egs --vocab-size=10003 - ark:- |'


# just a note on the unigram entropy of PTB training set:
# awk '{for (n=1;n<=NF;n++) { count[$n]++; } count["</s>"]++; } END{ tot_count=0; tot_entropy=0.0; for(k in count) tot_count += count[k];  for (k in count) { p = count[k]*1.0/tot_count; tot_entropy += p*log(p); }  print "entropy is " -tot_entropy; }' <data/text/ptb.txt
# 6.52933

# .. and entropy of bigrams:
# awk '{hist="<s>"; for (n=1;n<=NF;n++) { count[hist,$n]++; hist=$n; } count[hist,"</s>"]++; } END{ tot_count=0; tot_entropy=0.0; for(k in count) tot_count += count[k];  for (k in count) { p = count[k]*1.0/tot_count; tot_entropy += p*log(p); }  print "entropy is " -tot_entropy; }' <data/text/ptb.txt
# 10.7482
# in information theory, H(X) = H(Y) = 6.52, H(X,Y) = 10.7482, so H(Y | X) = 10.7482 - 6.52 = ***4.2282***, which
# is the entropy of the next symbol given the preceding symbol.  this gives a limit on the expected training
# objective given just a single word of context.
