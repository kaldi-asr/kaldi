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


stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=8
speed_perturb=true
use_pitch=true
use_ivector=true
global_extractor=exp/multi/nnet3/extractor
alidir=tri5_ali
suffix=
feat_suffix=_hires_mfcc # The feature suffix describing features used in multilingual training
                        # _hires_mfcc -> 40dim MFCC
                        # _hire_mfcc_pitch -> 40dim MFCC + pitch
                        # _hires_mfcc_pitch_bnf -> 40dim MFCC +pitch + BNF
print_interval=200
# corpora
# language list used for multilingual training
# The map for lang-name to its abreviation can be find in
# local/prepare_flp_langconf.sh
lang_list=(ASM CNT BNG HAI LAO  PSH  TAM  TGL  TUR  VTN  ZUL GRG)

# The language in this list decodes using Hybrid multilingual system.
decode_lang_list=(CNT ASM VTN TUR HAI LAO)

dir=exp/nnet3/multi_bnf
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 0 0"
frames_per_eg=8
avg_num_archives=4
cmd=queue.pl
num_epochs=4
bottleneck_dim=42
bnf_layer=5
ivector_suffix=_gb # if ivector_suffix = _gb, the iVector extracted using global iVector extractor
                   # trained on pooled data from all languages.
                   # Otherwise, it uses iVector extracted using local iVector extractor.

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

[ -f local.conf ] && . ./local.conf

num_lang=${#lang_list[@]}

echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

echo lang_list = ${lang_list[@]}

for lang in `seq 0 $[$num_lang-1]`; do
  for f in data/${lang_list[$lang]}/train/{feats.scp,text} exp/${lang_list[$lang]}/$alidir/ali.1.gz exp/${lang_list[$lang]}/$alidir/tree; do
   [ ! -f $f ] && echo "$0: no such file $f" && exit 1; 
if [ "$speed_perturb" == "true" ]; then
  suffix=${suffix}_sp
fi

if $use_pitch; then feat_suffix=${feat_suffix}_pitch ; fi
if $use_entropy;then feat_suffix=${feat_suffix}_entropy ; fi  
dir=${dir}${suffix}

# extract high resolution MFCC features for speed-perturbed data
# and extract alignment 
for lang in `seq 0 $[$num_lang-1]`; do
  local/nnet3/run_common_langs.sh --stage $stage \
    --speed-perturb $speed_perturb ${lang_list[$lang]} || exit;
done

# combine training data for all langs for training global i-vector extractor
if [ ! -f data/multi/train${suffix}_hires/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Pooling training data in data/multi${suffix}_hires on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/multi
  mkdir -p data/multi/train${suffix}_hires
  combine_lang_list=""
  for lang in `seq 0 $[$num_lang-1]`;do
    combine_lang_list="$combine_lang_list data/${lang_list[$lang]}/train${suffix}_hires"
  done
  utils/combine_data.sh data/multi/train${suffix}_hires $combine_lang_list
  utils/validate_data_dir.sh --no-feats data/multi/train${suffix}_hires
  touch data/multi/train${suffix}_hires/.done
fi
# If we do not use separate initial layer per language
# then we use ivector extractor trained on pooled data from all languages
# using an LDA+MLLT transform arbitrarily chonsed from single language.
if [ ! -f $global_extractor/.done ]; then
  echo "Generate global i-vector extractor"
  local/nnet3/run_shared_ivector_extractor.sh --global-extractor $global_extractor \
    --stage $stage $lda_mllt_lang || exit 1; 
  touch $global_extractor/.done
fi

# extract ivector for all languages.
for lang in `seq 0 $[$num_lang-1]`; do
  local/nnet3/run_ivector_common_langs.sh --stage $stage \
    --global-extractor $global_extractor \
    --speed-perturb $speed_perturb ${lang_list[$lang]} || exit;
done

# set num_leaves for all languages
for lang in `seq 0 $[$num_lang-1]`; do
  num_leaves=`tree-info exp/${lang_list[$lang]}/$alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  num_multiple_leaves="$num_multiple_leaves $num_leaves"

  multi_egs_dirs[$lang]=exp/${lang_list[$lang]}/nnet3/egs_v${avg_num_archives}${ivector_suffix}
  multi_ali_dirs[$lang]=exp/${lang_list[$lang]}/tri5_ali${suffix}

done

online_ivector_dir=
if $use_ivector; then
  online_ivector_dir=exp/${lang_list[0]}/nnet3/ivectors_train${suffix}${ivector_suffix}
fi

if [ -z "${online_ivector_dir}" ]; then
  echo "$0: Not using iVectors as no online-ivector-dir has been specified."
  ivector_dim=0
else
  ivector_dim=$(feat-to-dim scp:${online_ivector_dir}/ivector_online.scp -) || exit 1;
fi
feat_dim=`feat-to-dim scp:data/${lang_list[0]}/train${suffix}${feat_suffix}/feats.scp -`


if [ $stage -le 9 ]; then
  mkdir -p $dir/log
  echo "$0: creating neural net config for multilingual setups"
   # create the config files for nnet initialization
  $cmd $dir/log/make_config.log \
  python steps/nnet3/multilingual/make_tdnn_configs.py  \
    --splice-indexes "$splice_indexes"  \
    --feat-dim $feat_dim \
    --ivector-dim $ivector_dim  \
    --relu-dim 600 \
    --num-multiple-targets  "$num_multiple_leaves"  \
    --bottleneck-dim 42 --bottleneck-layer 5 \
   $dir/configs || exit 1;
  # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
  # matrix.  This first config just does any initial splicing that we do;
  # we do this as it's a convenient way to get the stats for the 'lda-like'
  # transform.
  $cmd $dir/log/nnet_init.log \
    nnet3-init --srand=-2 $dir/configs/init.config $dir/init.raw || exit 1;
fi

if [ $stage -le 10 ]; then
  echo "$0: generate separate egs directory per language for multilingual training."
  # sourcing the "vars" below sets
  #model_left_context=(something)
  #model_right_context=(something)
  #num_hidden_layers=(something)
  . $dir/configs/vars || exit 1;

  for lang in `seq 0 $[$num_lang-1]`; do
    egs_dir=${multi_egs_dirs[$lang]}
    ali_dir=${multi_ali_dirs[$lang]}
    data=data/${lang_list[$lang]}/train${suffix}${feat_suffix}
    num_frames=$(steps/nnet2/get_num_frames.sh $data)
    echo num_frames = $num_frames
    # sets samples_per_iter to have approximately 
    # same number of archives per language.
    samples_per_iter=$[$num_frames/($avg_num_archives*$frames_per_eg)]
    online_ivector_dir=
    if $use_ivector; then
      online_ivector_dir=exp/${lang_list[$lang]}/nnet3/ivectors_train${suffix}${ivector_suffix}
    fi
    if [ ! -d "$egs_dir" ]; then
      echo "$0: Generate egs for ${lang_list[$lang]}"
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
        utils/create_split_dir.pl \
         /export/b0{3,4,5,6}/$USER/kaldi-data/egs/${lang_list[$lang]}-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
      fi

      extra_opts=()
      [ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
      [ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
      [ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
      extra_opts+=(--left-context $model_left_context)
      extra_opts+=(--right-context $model_right_context)
      echo "$0: calling get_egs.sh"
      steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
          --samples-per-iter $samples_per_iter --stage $get_egs_stage \
          --cmd "$cmd" $egs_opts \
          --frames-per-eg $frames_per_eg \
          $data $ali_dir $egs_dir || exit 1;
      
    fi
  done
fi

if [ $stage -le 11 ]; then
  echo "$0: training mutilingual model."
  steps/nnet3/multilingual/train_tdnn.sh --cmd "$train_cmd" \
  --use-ivector $use_ivector --print-interval $print_interval \
  --num-epochs $num_epochs --cleanup false \
  --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
  --stage $train_stage \
  --initial-effective-lrate 0.0017 --final-effective-lrate 0.00017 \
  "${lang_list[@]}" "${multi_ali_dirs[@]}" "${multi_egs_dirs[@]}" \
  $dir || exit 1;

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
