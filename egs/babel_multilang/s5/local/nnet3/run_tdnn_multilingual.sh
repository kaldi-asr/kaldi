#!/usr/bin/env bash

# Copyright 2016 Pegah Ghahremani

# This script can be used for training multilingual setup using different
# languages (specifically babel languages) with no shared phones.
# It will generates separate egs directory for each dataset and combine them
# during training.
# In the new multilingual training setup, mini-batches of data corresponding to
# different languages are randomly combined to generate egs.*.scp files
# using steps/nnet3/multilingual/combine_egs.sh and generated egs.*.scp files used
# for multilingual training.
#
# For all languages, we share all except last hidden layer and there is separate final
# layer per language.
# The bottleneck layer can be added to the network structure using --bnf-dim option
#
# The script requires baseline PLP features and alignment (e.g. tri5_ali) for all languages.
# and it will generate 40dim MFCC + pitch features for all languages.
#
# The global iVector extractor trained using all languages by specifying
# --use-global-ivector-extractor and the iVectors are extracts for all languages.
#
# local.conf should exists (check README.txt), which contains configs for
# multilingual training such as lang_list as array of space-separated languages used
# for multilingual training.
#
echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e

remove_egs=false
cmd=queue.pl
srand=0
stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=8
speed_perturb=true
use_pitch=true  # if true, pitch feature used to train multilingual setup
use_pitch_ivector=false # if true, pitch feature used in ivector extraction.
use_ivector=true
megs_dir=
alidir=tri5_ali
ivector_extractor=  # If empty, the global iVector extractor trained on pooled data
                    # from all languages and iVectors are
                    # extracted using global ivector extractor and ivector_suffix ='_gb'.
                    # Otherwise this extractor is used to extract iVector and
                    # ivector_suffix = ''.

bnf_dim=           # If non-empty, the bottleneck layer with this dimension is added at two layers before softmax.
dir=exp/nnet3/multi_bnf

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

[ ! -f local.conf ] && echo 'the file local.conf does not exist! Read README.txt for more details.' && exit 1;
. local.conf || exit 1;

num_langs=${#lang_list[@]}
feat_suffix=_hires      # The feature suffix describing features used in
                        # multilingual training
                        # _hires -> 40dim MFCC
                        # _hires_pitch -> 40dim MFCC + pitch
                        # _hires_pitch_bnf -> 40dim MFCC +pitch + BNF

echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

for lang_index in `seq 0 $[$num_langs-1]`; do
  for f in data/${lang_list[$lang_index]}/train/{feats.scp,text} exp/${lang_list[$lang_index]}/$alidir/ali.1.gz exp/${lang_list[$lang_index]}/$alidir/tree; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
done

if [ "$speed_perturb" == "true" ]; then suffix=_sp; fi
dir=${dir}${suffix}

ivec_feat_suffix=${feat_suffix}
if $use_pitch; then feat_suffix=${feat_suffix}_pitch ; fi
if $use_pitch_ivector; then nnet3_affix=_pitch; ivec_feat_suffix=${feat_suffix}_pitch ; fi

for lang_index in `seq 0 $[$num_langs-1]`; do
  echo "$0: extract high resolution 40dim MFCC + pitch for speed-perturbed data "
  echo "and extract alignment."
  local/nnet3/run_common_langs.sh --stage $stage \
    --feat-suffix $feat_suffix \
    --use-pitch $use_pitch \
    --speed-perturb $speed_perturb ${lang_list[$lang_index]} || exit 1;
  if $use_pitch && ! $use_pitch_ivector; then
    echo "$0: select MFCC features for ivector extraction."
    featdir=data/${lang_list[$lang_index]}/train${suffix}${feat_suffix}
    ivec_featdir=data/${lang_list[$lang_index]}/train${suffix}${ivec_feat_suffix}
    mfcc_only_dim=`feat-to-dim scp:$featdir/feats.scp - | awk '{print $1-3}'`
    if [ ! -f $ivec_featdir/.done ]; then
      steps/select_feats.sh --cmd "$train_cmd" --nj 70 0-$[$mfcc_only_dim-1] \
        $featdir ${ivec_featdir} || exit 1;
      steps/compute_cmvn_stats.sh ${ivec_featdir} || exit 1;
      touch ${ivec_featdir}/.done || exit 1;
    fi
  fi
done

if $use_ivector; then
  ivector_suffix=""
  if [ -z "$ivector_extractor" ]; then
    mkdir -p data/multi
    global_extractor=exp/multi/nnet3${nnet3_affix}
    mkdir -p $global_extractor
    ivector_extractor=$global_extractor/extractor
    multi_data_dir_for_ivec=data/multi/train${suffix}${ivec_feat_suffix}
    ivector_suffix=_gb
    echo "$0: combine training data using all langs for training global i-vector extractor."
    if [ ! -f $multi_data_dir_for_ivec/.done ]; then
      echo ---------------------------------------------------------------------
      echo "Pooling training data in $multi_data_dir_for_ivec on" `date`
      echo ---------------------------------------------------------------------
      mkdir -p $multi_data_dir_for_ivec
      combine_lang_list=""
      for lang_index in `seq 0 $[$num_langs-1]`;do
        combine_lang_list="$combine_lang_list data/${lang_list[$lang_index]}/train${suffix}${ivec_feat_suffix}"
      done
      utils/combine_data.sh $multi_data_dir_for_ivec $combine_lang_list
      utils/validate_data_dir.sh --no-feats $multi_data_dir_for_ivec
      touch $multi_data_dir_for_ivec/.done
    fi
  fi
  if [ ! -f $global_extractor/extractor/.done ]; then
    if [ -z $lda_mllt_lang ]; then lda_mllt_lang=${lang_list[0]}; fi
    echo "$0: Generate global i-vector extractor on pooled data from all "
    echo "languages in $multi_data_dir_for_ivec, using an LDA+MLLT transform trained "
    echo "on ${lda_mllt_lang}."
    local/nnet3/run_shared_ivector_extractor.sh  \
      --suffix "$suffix" --nnet3-affix "$nnet3_affix" \
      --feat-suffix "$ivec_feat_suffix" \
      --stage $stage $lda_mllt_lang \
      $multi_data_dir_for_ivec $global_extractor || exit 1;
    touch $global_extractor/extractor/.done
  fi
  echo "$0: Extracts ivector for all languages using $global_extractor/extractor."
  for lang_index in `seq 0 $[$num_langs-1]`; do
    local/nnet3/extract_ivector_lang.sh --stage $stage \
      --train-set train${suffix}${ivec_feat_suffix} \
      --ivector-suffix "$ivector_suffix" \
      --nnet3-affix "$nnet3_affix" \
      ${lang_list[$lang_index]} \
      $ivector_extractor || exit;
  done
fi


for lang_index in `seq 0 $[$num_langs-1]`; do
  multi_data_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}${feat_suffix}
  multi_egs_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3${nnet3_affix}/egs${feat_suffix}${ivector_suffix}
  multi_ali_dirs[$lang_index]=exp/${lang_list[$lang_index]}/${alidir}${suffix}
  multi_ivector_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3${nnet3_affix}/ivectors_train${suffix}${ivec_feat_suffix}${ivector_suffix}
done


if $use_ivector; then
  ivector_dim=$(feat-to-dim scp:${multi_ivector_dirs[0]}/ivector_online.scp -) || exit 1;
else
  echo "$0: Not using iVectors in multilingual training."
  ivector_dim=0
fi
feat_dim=`feat-to-dim scp:${multi_data_dirs[0]}/feats.scp -`

if [ $stage -le 8 ]; then
  echo "$0: creating multilingual neural net configs using the xconfig parser";
  if [ -z $bnf_dim ]; then
    bnf_dim=1024
  fi
  mkdir -p $dir/configs
  ivector_node_xconfig=""
  ivector_to_append=""
  if $use_ivector; then
    ivector_node_xconfig="input dim=$ivector_dim name=ivector"
    ivector_to_append=", ReplaceIndex(ivector, t, 0)"
  fi
  cat <<EOF > $dir/configs/network.xconfig
  $ivector_node_xconfig
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2$ivector_to_append) dim=1024
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim
  # adding the layers for diffrent language's output
EOF
  # added separate outptut layer and softmax for all languages.
  for lang_index in `seq 0 $[$num_langs-1]`;do
    num_targets=`tree-info ${multi_ali_dirs[$lang_index]}/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;

    echo " relu-renorm-layer name=prefinal-affine-lang-${lang_index} input=tdnn_bn dim=1024"
    echo " output-layer name=output-${lang_index} dim=$num_targets max-change=1.5"
  done >> $dir/configs/network.xconfig

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ \
    --nnet-edits="rename-node old-name=output-0 new-name=output"
fi

if [ $stage -le 9 ]; then
  echo "$0: Generates separate egs dir per language for multilingual training."
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
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --left-context $model_left_context --right-context $model_right_context \
    $num_langs ${multi_data_dirs[@]} ${multi_ali_dirs[@]} ${multi_egs_dirs[@]} || exit 1;
fi

if [ -z $megs_dir ];then
  megs_dir=$dir/egs
fi

if [ $stage -le 10 ] && [ ! -z $megs_dir ]; then
  echo "$0: Generate multilingual egs dir using "
  echo "separate egs dirs for multilingual training."
  if [ ! -z "$lang2weight" ]; then
      egs_opts="--lang2weight '$lang2weight'"
  fi
  common_egs_dir="${multi_egs_dirs[@]} $megs_dir"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
    --cmd "$decode_cmd" \
    $num_langs ${common_egs_dir[@]} || exit 1;
fi

if [ $stage -le 11 ]; then
  common_ivec_dir=
  if $use_ivector;then
    common_ivec_dir=${multi_ivector_dirs[0]}
  fi
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir ${multi_data_dirs[0]} \
    --feat.online-ivector-dir "$common_ivec_dir" \
    --egs.dir $megs_dir \
    --use-dense-targets false \
    --targets-scp ${multi_ali_dirs[0]} \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
    lang_dir=$dir/${lang_list[$lang_index]}
    mkdir -p  $lang_dir
    echo "$0: rename output name for each lang to 'output' and "
    echo "add transition model."
    nnet3-copy --edits="rename-node old-name=output-$lang_index new-name=output" \
      $dir/final.raw - | \
      nnet3-am-init ${multi_ali_dirs[$lang_index]}/final.mdl - \
      $lang_dir/final.mdl || exit 1;
    cp $dir/cmvn_opts $lang_dir/cmvn_opts || exit 1;
    echo "$0: compute average posterior and readjust priors for language ${lang_list[$lang_index]}."
    steps/nnet3/adjust_priors.sh --cmd "$decode_cmd" \
      --use-gpu true \
      --iter final --use-raw-nnet false --use-gpu true \
      $lang_dir ${multi_egs_dirs[$lang_index]} || exit 1;
  done
fi

# decoding different languages
if [ $stage -le 13 ]; then
  num_decode_lang=${#decode_lang_list[@]}
  for lang_index in `seq 0 $[$num_decode_lang-1]`; do
    if [ ! -f $dir/${decode_lang_list[$lang_index]}/decode_dev10h.pem/.done ]; then
      echo "Decoding lang ${decode_lang_list[$lang_index]} using multilingual hybrid model $dir"
      local/nnet3/run_decode_lang.sh --use-ivector $use_ivector \
        --use-pitch-ivector $use_pitch_ivector --iter final_adj \
        --nnet3-affix "$nnet3_affix" \
        ${decode_lang_list[$lang_index]} $dir || exit 1;
      touch $dir/${decode_lang_list[$lang_index]}/decode_dev10h.pem/.done
    fi
  done
fi
