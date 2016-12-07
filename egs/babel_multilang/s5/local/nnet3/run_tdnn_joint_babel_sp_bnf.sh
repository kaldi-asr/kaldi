#!/bin/bash

# This script can be used for training multilingual setup using different
# languages (specifically babel languages) with no shared phones.
# It will generate separate egs directory for each dataset and combine them 
# during training.
# In the new multilingual training setup, mini-batches of data corresponding to 
# different languages are randomly sampled during training based on probability 
# distribution that reflects the relative frequency of the data from each language.

# For all languages, we share all the hidden layers and there is separate final
# layer per language.
# The bottleneck layer can be added to network structure.

# The script requires you to have baseline PLP features for all languages. 
# It generates 40dim MFCC + pitch features for all languages.

# The global iVector extractor is trained using all languages and the iVector
# extracts for all languages.

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e


cmd=queue.pl
stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=8
speed_perturb=true
use_pitch=true
global_extractor=exp/multi/nnet3/extractor
alidir=tri5_ali
suffix=
use_ivector=true
feat_suffix=_hires_mfcc # The feature suffix describing features used in multilingual training
                        # _hires_mfcc -> 40dim MFCC
                        # _hire_mfcc_pitch -> 40dim MFCC + pitch
                        # _hires_mfcc_pitch_bnf -> 40dim MFCC +pitch + BNF
# corpora
# language list used for multilingual training
# The map for lang-name to its abreviation can be find in
# local/prepare_lang_conf.sh
# e.g lang_list=(101-cantonese 102-assamese 103-bengali)
lang_list=
# The language in this list decodes using Hybrid multilingual system.
# e.g. decode_lang_list=(101-cantonese)
decode_lang_list=

dir=exp/nnet3/multi_bnf
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 0 0"

ivector_suffix=_gb # if ivector_suffix = _gb, the iVector extracted using global iVector extractor
                   # trained on pooled data from all languages.
                   # Otherwise, it uses iVector extracted using local iVector extractor.

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

[ -f local.conf ] && . ./local.conf

num_langs=${#lang_list[@]}

echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

echo "$0: lang_list = ${lang_list[@]}"

for lang_index in `seq 0 $[$num_langs-1]`; do
  for f in data/${lang_list[$lang_index]}/train/{feats.scp,text} exp/${lang_list[$lang_index]}/$alidir/ali.1.gz exp/${lang_list[$lang_index]}/$alidir/tree; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
done

if [ "$speed_perturb" == "true" ]; then
  suffix=${suffix}_sp
fi

if $use_pitch; then feat_suffix=${feat_suffix}_pitch ; fi
dir=${dir}${suffix}


# extract high resolution MFCC features for speed-perturbed data
# and extract alignment 
for lang_index in `seq 0 $[$num_langs-1]`; do
  echo "$0: extract 40dim MFCC + pitch for speed-perturbed data"
  local/nnet3/run_common_langs.sh --stage $stage \
    --speed-perturb $speed_perturb ${lang_list[$lang_index]} || exit;
done

# we use ivector extractor trained on pooled data from all languages
# using an LDA+MLLT transform arbitrarily chosen from single language.
if $use_ivector && [ ! -f $global_extractor/.done ]; then
  echo "$0: combine training data using all langs for training global i-vector extractor."
  if [ ! -f data/multi/train${suffix}_hires/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Pooling training data in data/multi${suffix}_hires on" `date`
    echo ---------------------------------------------------------------------
    mkdir -p data/multi
    mkdir -p data/multi/train${suffix}_hires
    combine_lang_list=""
    for lang_index in `seq 0 $[$num_langs-1]`;do
      combine_lang_list="$combine_lang_list data/${lang_list[$lang_index]}/train${suffix}_hires"
    done
    utils/combine_data.sh data/multi/train${suffix}_hires $combine_lang_list
    utils/validate_data_dir.sh --no-feats data/multi/train${suffix}_hires
    touch data/multi/train${suffix}_hires/.done
  fi

  echo "$0: Generate global i-vector extractor using data/multi"
  local/nnet3/run_shared_ivector_extractor.sh --global-extractor $global_extractor \
    --stage $stage ${lang_list[0]} || exit 1; 
  touch $global_extractor/.done

  echo "$0: Extract ivector for all languages."
  for lang_index in `seq 0 $[$num_langs-1]`; do
    local/nnet3/extract_ivector_lang.sh --stage $stage \
      --global-extractor $global_extractor \
      --train-set train$suffix ${lang_list[$lang_index]} || exit;
  done
fi


# set num_leaves for all languages
for lang_index in `seq 0 $[$num_langs-1]`; do
  num_leaves=`tree-info exp/${lang_list[$lang_index]}/$alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  num_multiple_leaves="$num_multiple_leaves $num_leaves"
  multi_data_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}${feat_suffix}
  multi_egs_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3/egs${ivector_suffix}
  multi_ali_dirs[$lang_index]=exp/${lang_list[$lang_index]}/tri5_ali${suffix}
  multi_ivector_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3/ivectors_train${suffix}${ivector_suffix} 
done

if $use_ivector; then
  ivector_dim=$(feat-to-dim scp:${multi_ivector_dirs[0]}/ivector_online.scp -) || exit 1;
  echo ivector-dim = $ivector_dim
else
  echo "$0: Not using iVectors in multilingual training."
  ivector_dim=0
fi

feat_dim=`feat-to-dim scp:${multi_data_dirs[0]}/feats.scp -`


if [ $stage -le 9 ]; then
  mkdir -p $dir/log
  echo "$0: creating neural net config for multilingual setups"
   # create the config files for nnet initialization
  $cmd $dir/log/make_config.log \
  python steps/nnet3/tdnn/make_configs.py  \
    --splice-indexes "$splice_indexes"  \
    --feat-dim $feat_dim \
    --ivector-dim $ivector_dim  \
    --relu-dim 600 \
    --num-multiple-targets  "$num_multiple_leaves"  \
    --bottleneck-dim 42 --bottleneck-layer 5 \
    --use-presoftmax-prior-scale false \
    --add-lda false \
   $dir/configs || exit 1;
  # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
  # matrix.  This first config just does any initial splicing that we do;
  # we do this as it's a convenient way to get the stats for the 'lda-like'
  # transform.
  $cmd $dir/log/nnet_init.log \
    nnet3-init --srand=-2 $dir/configs/init.config $dir/init.raw || exit 1;
fi

if [ $stage -le 10 ]; then
  echo "$0: Generate separate egs dir per language for multilingual training."
  # sourcing the "vars" below sets
  #model_left_context=(something)
  #model_right_context=(something)
  #num_hidden_layers=(something)
  . $dir/configs/vars || exit 1;
  

  ivec="${multi_ivector_dirs[@]}"
  if $use_ivector; then
    ivector_opts=(--online-multi-ivector-dirs "$ivec")
  fi
  local/nnet3/prepare_multilingual_egs.sh --cmd "$decode_cmd" \
    "${ivector_opts[@]}" \
    --left-context $model_left_context --right-context $model_right_context \
    --samples-per-iter 400000 \
    $num_langs ${multi_data_dirs[@]} ${multi_ali_dirs[@]} ${multi_egs_dirs[@]} || exit 1;
fi

if [ $stage -le 11 ]; then
  echo "$0: training mutilingual model."
  common_egs_dir="${multi_egs_dirs[@]} $dir/egs"
  echo common_egs_dir = $common_egs_dir
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --use-dense-target false \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --feat-dir ${multi_data_dirs[0]} \
    --feat.online-ivector-dir ${multi_ivector_dirs[0]} \
    --egs.dir "${common_egs_dir[@]}" \
    --cleanup.remove-egs false \
    --cleanup.preserve-model-interval 20 \
    --use-gpu true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

# decoding different languages
if [ $stage -le 12 ]; then
  num_decode_lang=${#decode_lang_list[@]}
  (
  for lang in `seq 0 $[$num_decode_lang-1]`; do
    if [ ! -f $dir/${decode_lang_list[$lang]}/decode_dev10h.pem/.done ]; then 
      cp $dir/cmvn_opts $dir/${decode_lang_list[$lang]}/.
      echo "Decoding lang ${decode_lang_list[$lang]} using multilingual hybrid model $dir"
      run-4-anydecode-langs.sh --use-ivector $use_ivector --nnet3-dir $dir ${decode_lang_list[$lang]} || exit 1;
      touch $dir/${decode_lang_list[$lang]}/decode_dev10h.pem/.done
    fi
  done
  wait
  )
fi
