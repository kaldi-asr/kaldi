#!/bin/bash

# This script generates separate egs directory for each input 
# language in multilingual setup, which contains both egs.*.ark and egs.*.scp.
#
# Then it uses separate egs directory for each language, 
# to generate egs.*.scp file where the scp points to
# original archive for each egs directory.
# egs.*.scp is randomized w.r.t language id.
#
# Also it generates egs.JOB.scp, output.JOB.scp and weight.JOB.scp,
# where output file contains language-id for each example
# and weight file contains weights for scaling output posterior 
# for each example w.r.t input language.
#
# This script will generally be called from nnet training script
# and combine training examples used to train multilingual neural net.

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e


# Begin configuration section
cmd=run.pl
stage=0
get_egs_stage=0
left_context=13
right_context=9
online_ivector_dir=     # can be used if we are including speaker information as iVectors.
samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the
                        # entire data.
# Configuration to allocate egs
minibatch_size=512
num_archives=100
num_jobs=10
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 4 ]; then
  echo "Usage: $0 [opts] num-input-langs <data-dir-per-lang> <ali-dir-per-lang> <egs-dir-per-lang> <multilingual-egs-dir>"
  echo " e.g.: $0 2 data/lang1/train data/lang2/train "
       " exp/lang1/tri5_ali exp/lang2/tri5_ali exp/lang1/nnet3/lang1 exp/lang2/nnet3/lang2 exp/multi/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --num-jobs <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --samples-per-iter <#samples;400000>             # Target number of egs per archive (option is badly named)"
  echo "  --frames-per-eg <frames;8>                       # number of frames per eg on disk"
  echo "  --left-context <width;4>                         # Number of frames on left side to append for feature input"
  echo "  --right-context <width;4>                        # Number of frames on right side to append for feature input"
  echo "  --num-frames-diagnostic <#frames;4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames;10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

num_lang=$1
shift
args=("$@")
megs_dir=${args[-1]}
num_lang_args=$[$[${#args[@]}-1]/3] 

if [ ${#args[@]} != $[$num_lang*3+1] ]; then
  echo "$0: num of input dirs provided for all langs is not compatible with num-langs in input." && exit 1;
fi

# read input data, ali and egs dir per lang
for l in `seq 0 $[$num_lang-1]`; do
  multi_data_dirs[$l]=${args[$l]}
  multi_ali_dirs[$l]=${args[$l+$num_lang]}
  multi_egs_dirs[$l]=${args[$l+2*$num_lang]}
done

if [ $stage -le 0 ];then
  echo "$0: Generate separate egs directory per language for multilingual training."
  echo num_langs = $num_lang
  for lang in `seq 0 $[$num_lang-1]`; do
    data=${multi_data_dirs[$lang]} 
    ali_dir=${multi_ali_dirs[$lang]}
    egs_dir=${multi_egs_dirs[$lang]}
    online_ivector_dir=
    if [ ! -z "$multi_ivector_dirs" ]; then
      online_ivector_dir=${multi_ivector_dirs[$lang]}
    fi
    #if [ ! -d "$egs_dir" ]; then
    if true; then
      echo "$0: Generate egs for ${lang_list[$lang]}"
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
        utils/create_split_dir.pl \
         /export/b0{3,4,5,6}/$USER/kaldi-data/egs/${lang_list[$lang]}-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
      fi

      extra_opts=()
      [ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
      [ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
      extra_opts+=(--left-context $left_context)
      extra_opts+=(--right-context $right_context)
      echo "$0: calling get_egs.sh"
      steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
          --samples-per-iter $samples_per_iter --stage $get_egs_stage \
          --cmd "$cmd" $egs_opts \
          --generate-egs-scp true \
          $data $ali_dir $egs_dir || exit 1;
      
    fi
  done
fi

if [ $stage -le 1 ]; then
  echo "$0: allocating multilingual examples for training."

  # concatenate egs.scp from different lang's egs directory.
  rm -rf $megs_dir/lang2len.train
  for l in $(seq 0 $[num_lang-1]); do
    echo dir = ${multi_egs_dirs[$l]}
    len=`wc -l ${multi_egs_dirs[$l]}/egs.scp | cut -d" " -f1` 
    echo "$l $len" >> $megs_dir/lang2len.train
  done 

  # Generate range for generating egs.JOB.scp for multilingual setup.
  # where each line is interpreted as follows:
  # <source-language> <absolute-archive-index> <local-scp-line> <num-examples>
  # e.g.
  # lang1 0 0 256
  # lang2 1 256 256
  $cmd $megs_dir/log/allocate_multilingual_examples_train.log \
  python steps/nnet3/multilingual/allocate_multilingual_examples.py \
      --num-archives $num_archives \
      --num-jobs $num_jobs --minibatch-size $minibatch_size \
      $megs_dir/lang2len.train $megs_dir || exit 1;
  
fi



if [ $stage -le 2 ]; then
  echo "$0: Generating egs.job.scp for training using ranges.job"
  (
    for j in $(seq $num_jobs); do 
      $cmd $megs_dir/log/generate_scp.$j.log \
        steps/nnet3/multilingual/extract_scp.sh $num_lang "${multi_egs_dirs[@]}" $megs_dir/temp/ranges.$j $megs_dir/egs.$j.scp || touch $megs_dir/.error & 
    done
    wait
  )
  echo "$0: allocating multilingaul example for training subset"
fi
