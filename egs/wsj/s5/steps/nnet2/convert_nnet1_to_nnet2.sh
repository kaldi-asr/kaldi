#!/bin/bash

# Copyright 2014    Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script converts nnet1 into nnet2 models.
# Note, it doesn't support all possible types of nnet1 models.

# Begin configuration section
cleanup=true
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <src-nnet1-dir> <dest-nnet2-dir>"
  echo "e.g.: $0 exp/dnn4b_pretrain-dbn_dnn exp/dnn4b_nnet2"
  exit 1;
fi

src=$1
dir=$2

mkdir -p $dir/log || exit 1;

for f in $src/final.mdl $src/final.feature_transform $src/ali_train_pdf.counts; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

cp $src/phones.txt $dir 2>/dev/null

$cmd $dir/log/convert_feature_transform.log \
  nnet1-to-raw-nnet $src/final.feature_transform $dir/0.raw || exit 1;


if [ -f $src/final.nnet ]; then
  echo "$0: $src/final.nnet exists, using it as input."
  $cmd $dir/log/convert_model.log \
    nnet1-to-raw-nnet $src/final.nnet $dir/1.raw || exit 1;
elif [ -f $src/final.dbn ]; then
  echo "$0: $src/final.dbn exists, using it as input."
  num_leaves=$(am-info $src/final.mdl | grep -w pdfs | awk '{print $NF}') || exit 1;
  dbn_output_dim=$(nnet-info exp/dnn4b_pretrain-dbn/6.dbn  | grep component | tail -n 1 | sed s:,::g | awk '{print $NF}') || exit 1;
  [ -z "$dbn_output_dim" ] && exit 1;
  
  cat > $dir/final_layer.conf <<EOF
AffineComponent input-dim=$dbn_output_dim output-dim=$num_leaves learning-rate=0.001
SoftmaxComponent dim=$num_leaves
EOF
  $cmd $dir/log/convert_model.log \
    nnet1-to-raw-nnet $src/final.dbn - \| \
    raw-nnet-concat - "raw-nnet-init $dir/final_layer.conf -|" $dir/1.raw || exit 1;
else
  echo "$0: expected either $src/final.nnet or $src/final.dbn to exist"
fi

$cmd $dir/log/append_model.log \
  raw-nnet-concat $dir/0.raw $dir/1.raw $dir/concat.raw || exit 1;

$cmd $dir/log/init_model.log \
  nnet-am-init $src/final.mdl $dir/concat.raw $dir/final_noprior.mdl || exit 1;

$cmd $dir/log/set_priors.log \
  nnet-adjust-priors $dir/final_noprior.mdl $src/ali_train_pdf.counts $dir/final.mdl || exit 1;

if $cleanup; then
  rm $dir/0.raw $dir/1.raw $dir/concat.raw $dir/final_noprior.mdl
fi
