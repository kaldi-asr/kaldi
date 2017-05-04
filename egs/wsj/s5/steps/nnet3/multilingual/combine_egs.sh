#!/bin/bash
#
# This script generates examples for multilingual training of neural network
# using separate input egs dir per language as input.
# This scripts produces 3 sets of files --
# egs.*.scp, egs.output.*.ark, egs.weight.*.ark
#
# egs.*.scp are the SCP files of the training examples.
# egs.weight.*.ark map from the key of the example to the language-specific
# weight of that example.
# egs.output.*.ark map from the key of the example to the name of
# the output-node in the neural net for that specific language, e.g.
# 'output-2'.
#
# Begin configuration section.
cmd=run.pl
minibatch_size=512      # it is the number of consecutive egs that we take from 
                        # each source, and it only affects the locality of disk 
                        # access. This does not have to be the actual minibatch size;
num_jobs=10             # helps for better randomness across languages
                        # per archive.
samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the
                        # entire data.
lang2weight=            # array of weights one per input languge to scale example's output
                        # w.r.t its input language during training.
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
  echo "$0: num of input example dirs provided is not compatible with num_langs $num_langs."
  echo "Usage:$0 [opts] <num-input-langs,N> <lang1-egs-dir> ...<langN-egs-dir> <multilingual-egs-dir>"
  echo "Usage:$0 [opts] 2 exp/lang1/egs exp/lang2/egs exp/multi/egs"
  exit 1;
fi

required="egs.scp combine.scp train_diagnostic.scp valid_diagnostic.scp"
train_scp_list=
train_diagnostic_scp_list=
valid_diagnostic_scp_list=
combine_scp_list=

# read paramter from $egs_dir[0]/info and cmvn_opts
# to write in multilingual egs_dir.
check_params="info/feat_dim info/ivector_dim info/left_context info/right_context info/frames_per_eg info/final.ie.id cmvn_opts"
for param in $check_params; do
  cat ${args[0]}/$param > $megs_dir/$param || exit 1;
done
cat ${args[0]}/cmvn_opts > $megs_dir/cmvn_opts || exit 1; # caution: the top-level nnet training
                                                          # script should copy this to its
                                                          # own dir.
for lang in $(seq 0 $[$num_langs-1]);do
  multi_egs_dir[$lang]=${args[$lang]}
  for f in $required; do
    if [ ! -f ${multi_egs_dir[$lang]}/$f ]; then
      echo "$0: no such file ${multi_egs_dir[$lang]}/$f." && exit 1;
    fi
  done
  train_scp_list="$train_scp_list ${args[$lang]}/egs.scp"
  train_diagnostic_scp_list="$train_diagnostic_scp_list ${args[$lang]}/train_diagnostic.scp"
  valid_diagnostic_scp_list="$valid_diagnostic_scp_list ${args[$lang]}/valid_diagnostic.scp"
  combine_scp_list="$combine_scp_list ${args[$lang]}/combine.scp"

  # check parameter dimension to be the same in all egs dirs
  for f in $check_params; do
    if [ -f $megs_dir/$f ] && [ -f ${multi_egs_dir[$lang]}/$f ]; then
      f1=$(cat $megs_dir/$f)
      f2=$(cat ${multi_egs_dir[$lang]}/$f)
      if [ "$f1" != "$f2" ]  ; then
        echo "$0: mismatch for $f in $megs_dir vs. ${multi_egs_dir[$lang]}($f1 vs. $f2)."
        exit 1;
      fi
    else
      echo "$0: file $f does not exits in $megs_dir or ${multi_egs_dir[$lang]}/$f ."
    fi
  done
done

if [ $stage -le 0 ]; then
  echo "$0: allocating multilingual examples for training."
  if [ ! -z "$lang2weight" ]; then
    egs_opt="--lang2weight '$lang2weight'"
  fi
  # Generate egs.*.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_train.log \
  steps/nnet3/multilingual/allocate_multilingual_examples.py $egs_opt \
      --minibatch-size $minibatch_size \
      --samples-per-iter $samples_per_iter \
      $train_scp_list $megs_dir || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combine combine.scp examples from all langs in $megs_dir/combine.scp."
  # Generate combine.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_combine.log \
  steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --random-lang false \
      --max-archives 1 --num-jobs 1 \
      --minibatch-size $minibatch_size \
      --egs-prefix "combine." \
      $combine_scp_list $megs_dir || exit 1;

  echo "$0: combine train_diagnostic.scp examples from all langs in $megs_dir/train_diagnostic.scp."
  # Generate train_diagnostic.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_train_diagnostic.log \
  steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --random-lang false \
      --max-archives 1 --num-jobs 1 \
      --minibatch-size $minibatch_size \
      --egs-prefix "train_diagnostic." \
      $train_diagnostic_scp_list $megs_dir || exit 1;


  echo "$0: combine valid_diagnostic.scp examples from all langs in $megs_dir/valid_diagnostic.scp."
  # Generate valid_diagnostic.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_valid_diagnostic.log \
  steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --random-lang false --max-archives 1 --num-jobs 1\
      --minibatch-size $minibatch_size \
      --egs-prefix "valid_diagnostic." \
      $valid_diagnostic_scp_list $megs_dir || exit 1;

fi
for egs_type in combine train_diagnostic valid_diagnostic; do
  mv $megs_dir/${egs_type}.output.1.ark $megs_dir/${egs_type}.output.ark || exit 1;
  mv $megs_dir/${egs_type}.weight.1.ark $megs_dir/${egs_type}.weight.ark || exit 1;
  mv $megs_dir/${egs_type}.1.scp $megs_dir/${egs_type}.scp || exit 1;
done
mv $megs_dir/info/egs.num_archives $megs_dir/info/num_archives || exit 1;
mv $megs_dir/info/egs.num_tasks $megs_dir/info/num_tasks || exit 1;
echo "$0: Finished preparing multilingual training example."
