#!/bin/bash
#
# This script uses separate input egs directory for each language as input, 
# to generate egs.*.scp files in multilingual egs directory
# where the scp line points to the original archive for each egs directory.
# $megs/egs.*.scp is randomized w.r.t language id.
#
# Also this script generates egs.JOB.scp, output.JOB.scp and weight.JOB.scp,
# where output file contains language-id for each example
# and weight file contains weights for scaling output posterior 
# for each example w.r.t input language.
#

set -e 
set -o pipefail
set -u

# Begin configuration section.
cmd=run.pl
minibatch_size=512      # multiple of minibatch used during training.
minibatch_size=
num_jobs=10             # This can be set to max number of jobs to run in parallel;
                        # Helps for better randomness across languages
                        # per archive.
samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the
                        # entire data.
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

num_langs=$1
shift 1
args=("$@")
megs_dir=${args[-1]} # multilingual directory
mkdir -p $megs_dir
mkdir -p $megs_dir/info

if [ ${#args[@]} != $[$num_langs+1] ]; then
  echo "$0: Number of input example dirs provided is not compatible with num_langs $num_langs."
  echo "Usage:$0 [opts] <num-input-langs,N> <lang1-egs-dir> ...<langN-egs-dir> <multilingual-egs-dir>"
  echo "Usage:$0 [opts] 2 exp/lang1/egs exp/lang2/egs exp/multi/egs"
  exit 1;
fi

required_files="egs.scp combine.egs.scp train_diagnostic.egs.scp valid_diagnostic.egs.scp"
train_scp_list=
train_diagnostic_scp_list=
valid_diagnostic_scp_list=
combine_scp_list=

# copy paramters from $egs_dir[0]/info
# into multilingual dir egs_dir/info

params_to_check="feat_dim ivector_dim left_context right_context frames_per_eg"
for param in $params_to_check; do
  cat ${args[0]}/info/$param > $megs_dir/info/$param || exit 1;
done

for lang in $(seq 0 $[$num_langs-1]);do
  multi_egs_dir[$lang]=${args[$lang]}
  echo "arg[$lang] = ${args[$lang]}"
  for f in $required_files; do
    if [ ! -f ${multi_egs_dir[$lang]}/$f ]; then
      echo "$0: no such a file ${multi_egs_dir[$lang]}/$f." && exit 1;
    fi
  done
  train_scp_list="$train_scp_list ${args[$lang]}/egs.scp"
  train_diagnostic_scp_list="$train_diagnostic_scp_list ${args[$lang]}/train_diagnostic.egs.scp"
  valid_diagnostic_scp_list="$valid_diagnostic_scp_list ${args[$lang]}/valid_diagnostic.egs.scp"
  combine_scp_list="$combine_scp_list ${args[$lang]}/combine.egs.scp"

  # check parameter dimension to be the same in all egs dirs
  for f in $params_to_check; do
    f1=`cat $megs_dir/info/$param`;
    f2=`cat ${multi_egs_dir[$lang]}/info/$f`;
    if [ $f1 != $f1 ]; then
      echo "$0: mismatch in dimension for $f parameter in ${multi_egs_dir[$lang]}." 
      exit 1;
    fi
  done
done

cp ${multi_egs_dir[$lang]}/cmvn_opts $megs_dir

if [ $stage -le 0 ]; then
  echo "$0: allocating multilingual examples for training."
  # Generate egs.*.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_train.log \
  python steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --minibatch-size $minibatch_size \
      --samples-per-iter $samples_per_iter \
      $num_langs "$train_scp_list" $megs_dir || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combine combine.egs.scp examples from all langs in $megs_dir/combine.egs.scp."
  # Generate combine.egs.scp for multilingual setup. 
  $cmd $megs_dir/log/allocate_multilingual_examples_combine.log \
  python steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --random-lang false \
      --max-archives 1 --num-jobs 1 \
      --minibatch-size $minibatch_size \
      --prefix "combine." \
      $num_langs "$combine_scp_list" $megs_dir || exit 1;
  
  echo "$0: combine train_diagnostic.egs.scp examples from all langs in $megs_dir/train_diagnostic.egs.scp."
  # Generate train_diagnostic.egs.scp for multilingual setup. 
  $cmd $megs_dir/log/allocate_multilingual_examples_train_diagnostic.log \
  python steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --random-lang false \
      --max-archives 1 --num-jobs 1 \
      --minibatch-size $minibatch_size \
      --prefix "train_diagnostic." \
      $num_langs "$train_diagnostic_scp_list" $megs_dir || exit 1;

      
  echo "$0: combine valid_diagnostic.egs.scp examples from all langs in $megs_dir/valid_diagnostic.egs.scp."
  # Generate valid_diagnostic.egs.scp for multilingual setup. 
  $cmd $megs_dir/log/allocate_multilingual_examples_valid_diagnostic.log \
  python steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --random-lang false --max-archives 1 --num-jobs 1\
      --minibatch-size $minibatch_size \
      --prefix "valid_diagnostic." \
      $num_langs "$valid_diagnostic_scp_list" $megs_dir || exit 1;
   
fi

