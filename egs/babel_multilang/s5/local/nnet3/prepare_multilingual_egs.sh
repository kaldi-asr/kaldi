#!/bin/bash
#
# This script generates separate egs directory for each input
# language in multilingual setup, which contains both egs.*.ark and egs.*.scp.
#
# This script will generally be called from nnet3 multilingual training script.

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e


# Begin configuration section
cmd=
stage=0
left_context=13
right_context=9
online_multi_ivector_dirs=     # list of iVector dir for all languages
                              # can be used if we are including speaker information as iVectors.
                              # e.g. "exp/lang1/train-ivector exp/lang2/train-ivector"
samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the
                        # entire data.
# Configuration to allocate egs
minibatch_size=512
num_archives=100
num_jobs=10
cmvn_opts=
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 4 ]; then
  echo "Usage: $0 [opts] N <data-dir1> .. <data-dirN> <ali-dir1> .. <ali-dirN>"
  echo " <egs-out1> .. <egs-outN>"
  echo " e.g.: $0 2 data/lang1/train data/lang2/train exp/lang1/tri5_ali"
  echo " exp/lang2/tri5_ali exp/lang1/nnet3/egs exp/lang2/nnet3/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --num-jobs <nj>                                  # The maximum number of jobs you want to run in"
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

if [ ${#args[@]} != $[$num_lang*3] ]; then
  echo "$0: num of input dirs provided for all langs is not compatible with num-langs in input." && exit 1;
fi

# read input data, ali and egs dir per lang
for l in `seq 0 $[$num_lang-1]`; do
  multi_data_dirs[$l]=${args[$l]}
  multi_ali_dirs[$l]=${args[$l+$num_lang]}
  multi_egs_dirs[$l]=${args[$l+2*$num_lang]}
done

echo "$0: Generate separate egs directory per language for multilingual training."
online_multi_ivector_dirs=(${online_multi_ivector_dirs[@]})
for lang_index in `seq 0 $[$num_lang-1]`; do
  data=${multi_data_dirs[$lang_index]}
  ali_dir=${multi_ali_dirs[$lang_index]}
  egs_dir=${multi_egs_dirs[$lang_index]}
  online_ivector_dir=
  if [ ! -z "${online_multi_ivector_dirs[$lang_index]}" ]; then
    online_ivector_dir=${online_multi_ivector_dirs[$lang_index]}
  fi
  echo online_ivector_dir = $online_ivector_dir
  if [ ! -d "$egs_dir" ]; then
    echo "$0: Generate egs for ${lang_list[$lang_index]}"
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{3,4,5,6}/$USER/kaldi-data/egs/${lang_list[$lang_index]}-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
    fi

    extra_opts=()
    [ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
    [ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
    extra_opts+=(--left-context $left_context)
    extra_opts+=(--right-context $right_context)
    echo "$0: calling get_egs.sh"
    steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
        --samples-per-iter $samples_per_iter --stage $stage \
        --cmd "$cmd" $egs_opts \
        --generate-egs-scp true \
        $data $ali_dir $egs_dir || exit 1;

  fi
done

