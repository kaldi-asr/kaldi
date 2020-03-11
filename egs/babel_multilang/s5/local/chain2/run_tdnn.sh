#!/bin/bash
# chain2 recipe for monolingual systems for BABEL
# Copyright 2016 Pegah Ghahremani
# Copyright 2020 Srikanth Madikeri (Idiap Research Institute)

# This script is used to train multilingual LF-MMI system with a multi-task training
# setup.

# local.conf should exists (check README.txt), which contains configs for
# multilingual training such as lang_list as array of space-separated languages used
# for multilingual training.

set -e -o pipefail

remove_egs=false
cmd=queue.pl
srand=-1
stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10

speed_perturb=true
use_pitch=true  # if true, pitch feature used to train multilingual setup
use_pitch_ivector=false # if true, pitch feature used in ivector extraction.
use_ivector=true
megs_dir=
alidir=tri5_ali
stage=-1
nj=30
train_set=train
gmm=tri5  # the gmm for the target data
langdir=data/lang
num_threads_ubm=1
nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
feat_suffix=_hires      

label_delay=5
frame_subsampling_factor=3
xent_regularize=0.01
max_param_change=2.0
num_jobs_initial=2
num_jobs_final=12
initial_effective_lrate=0.001
final_effective_lrate=0.0001
num_jobs_initial=2
num_jobs_final=8
chunk_width=150
extra_left_context=50
extra_right_context=0
common_egs_dir=  # you can set this to use previously dumped egs.
langconf=local.conf

speed_perturb=true
global_extractor=exp/multi/nnet3/extractor
dir=exp/chain2${nnet3_affix}/tdnn${tdnn_affix}_multi

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1;
. $langconf || exit 1;

[ ! -f local.conf ] && echo 'the file local.conf does not exist!' && exit 1;
. local.conf || exit 1;

suffix=
if $speed_perturb; then
  suffix=_sp
fi

num_langs=${#lang_list[@]}
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
        lang_name=${lang_list[$lang_index]}
        utils/copy_data_dir.sh --spk-prefix ${lang_name}- --utt-prefix ${lang_name}- data/${lang_list[$lang_index]}/train${suffix}${ivec_feat_suffix} data/${lang_list[$lang_index]}/train${suffix}${ivec_feat_suffix}_prefixed || exit 1
        combine_lang_list="$combine_lang_list data/${lang_list[$lang_index]}/train${suffix}${ivec_feat_suffix}_prefixed"
      done
      utils/combine_data.sh $multi_data_dir_for_ivec $combine_lang_list
      utils/validate_data_dir.sh --no-feats $multi_data_dir_for_ivec
      for lang_index in `seq 0 $[$num_langs-1]`;do
        lang_name=${lang_list[$lang_index]}
        rm -r data/${lang_list[$lang_index]}/train${suffix}${ivec_feat_suffix}_prefixed
      done
      touch $multi_data_dir_for_ivec/.done
    fi
  fi
  if [ ! -f $global_extractor/extractor/.done ]; then
    local/nnet3/run_shared_ivector_extractor.sh  \
      --suffix "$suffix" --nnet3-affix "$nnet3_affix" \
      --feat-suffix "$ivec_feat_suffix" \
      --ivector-transform-type pca \
      --stage $stage multi \
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

dir_basename=`basename $dir`
for lang_index in `seq 0 $[$num_langs-1]`; do
    lang_name=${lang_list[$lang_index]}
  multi_lores_data_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}
  multi_data_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}${feat_suffix}
  multi_egs_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3${nnet3_affix}/egs${feat_suffix}${ivector_suffix}
  multi_ali_dirs[$lang_index]=exp/${lang_list[$lang_index]}/${alidir}${suffix}
  multi_ivector_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3${nnet3_affix}/ivectors_train${suffix}${ivec_feat_suffix}${ivector_suffix}
  multi_ali_treedirs[$lang_index]=exp/${lang_list[$lang_index]}/tree${tree_affix}
  multi_ali_latdirs[$lang_index]=exp/${lang_list[$lang_index]}/chain/${gmm}_train${suffix}_lats
  multi_lang[$lang_index]=data/${lang_list[$lang_index]}/lang
  multi_lfmmi_lang[$lang_index]=data/${lang_list[$lang_index]}/lang_chain
  multi_gmm_dir[$lang_index]=exp/${lang_list[$lang_index]}/$gmm
  multi_chain_dir[$lang_index]=exp/${lang_list[$lang_index]}/chain/$dir_basename
done

if $use_ivector; then
  ivector_dim=$(feat-to-dim scp:${multi_ivector_dirs[0]}/ivector_online.scp -) || exit 1;
else
  echo "$0: Not using iVectors in multilingual training."
  ivector_dim=0
fi
feat_dim=`feat-to-dim scp:${multi_data_dirs[0]}/feats.scp -`

if [ $stage -le 8 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
      lang_name=${lang_list[$lang_index]}
      if [ -d ${multi_lfmmi_lang[$lang_index]} ]; then
        if [ ${multi_lfmmi_lang[$lang_index]}/L.fst -nt ${multi_lang[$lang_index]}/L.fst ]; then
          echo "$0: ${multi_lfmmi_lang[$lang_index]} already exists, not overwriting it; continuing"
        else
          echo "$0: ${multi_lfmmi_lang[$lang_index]} already exists and seems to be older than ${multi_lang[$lang_index]}..."
          echo " ... not sure what to do.  Exiting."
          exit 1;
        fi
      else
        echo "$0: creating lang directory with one state per phone."
        cp -r ${multi_lang[$lang_index]}/ ${multi_lfmmi_lang[$lang_index]} # trailing slash makes sure soft links are copied
        silphonelist=$(cat ${multi_lfmmi_lang[$lang_index]}/phones/silence.csl) || exit 1;
        nonsilphonelist=$(cat ${multi_lfmmi_lang[$lang_index]}/phones/nonsilence.csl) || exit 1;
        # Use our special topology... note that later on may have to tune this
        # topology.
        steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >${multi_lfmmi_lang[$lang_index]}/topo
      fi
  done
fi

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  for lang_index in `seq 0 $[$num_langs-1]`;do
      langdir=${multi_lang[$lang_index]}
      lores_train_data_dir=${multi_lores_data_dirs[$lang_index]}
      gmm_dir=${multi_gmm_dir[$lang_index]}
      lat_dir=${multi_ali_latdirs[$lang_index]}

      steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
        $langdir $gmm_dir $lat_dir
      rm $lat_dir/fsts.*.gz # save space
      exit
  done
fi 

if [ $stage -le 10 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
      lang_name=${lang_list[$lang_index]}
      echo "$0: Building tree for $lang_name"

      tree_dir=${multi_ali_treedirs[$lang_index]}
      ali_dir=${multi_ali_dirs[$lang_index]}
      lores_train_data_dir=${multi_lores_data_dirs[$lang_index]}
      lang_dir=${multi_lfmmi_lang[$lang_index]}
      if [ -f $tree_dir/final.mdl -a -f $tree_dir/tree ]; then
        echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
        continue
      fi
      steps/nnet3/chain/build_tree.sh --frame-subsampling-factor $frame_subsampling_factor \
          --context-opts "--context-width=2 --central-position=1" \
          --leftmost-questions-truncate -1 \
          --cmd "$train_cmd" 4000 ${lores_train_data_dir} $lang_dir $ali_dir $tree_dir
  done
fi

if [ $stage -le 11 ]; then
  echo "$0: creating multilingual neural net configs using the xconfig parser";
  if [ -z $bnf_dim ]; then
    bnf_dim=80
  fi
  mkdir -p $dir/configs
  ivector_node_xconfig=""
  ivector_to_append=""
  if $use_ivector; then
    ivector_node_xconfig="input dim=$ivector_dim name=ivector"
    ivector_to_append=", ReplaceIndex(ivector, t, 0)"
  fi
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  dummy_tree_dir=${multi_ali_treedirs[0]}
  num_targets=`tree-info $dummy_tree_dir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input
  $ivector_node_xconfig

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2$ivector_to_append) dim=450
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=450
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn7 input=Append(-6,-3,0) dim=450
  #relu-batchnorm-layer name=tdnn_bn dim=$bnf_dim
  # adding the layers for diffrent language's output
  # dummy output node
  output-layer name=output dim=$num_targets max-change=1.5 include-log-softmax=false
  output-layer name=output-xent input=tdnn7 dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  # added separate outptut layer and softmax for all languages.
  for lang_index in `seq 0 $[$num_langs-1]`;do
    tree_dir=${multi_ali_treedirs[$lang_index]}
    num_targets=`tree-info $tree_dir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;

    lang_name=${lang_list[${lang_index}]}
    #echo "relu-renorm-layer name=prefinal-affine-lang-${lang_name} input=tdnn7 dim=450 target-rms=0.5"
    echo "output-layer name=output-${lang_name} dim=$num_targets input=tdnn7  max-change=1.5 include-log-softmax=false"
    echo "output-layer name=output-${lang_name}-xent input=tdnn7 dim=$num_targets  learning-rate-factor=$learning_rate_factor max-change=1.5"
  done >> $dir/configs/network.xconfig

  lang_name=${lang_list[0]}
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ 
fi

init_info=$dir/init/info.txt
if [ $stage -le 12 ]; then
  if [ ! -f $dir/configs/ref.raw ]; then
      echo "Expected $dir/configs/ref.raw to exist"
      exit
  fi
  mkdir  -p $dir/init
  nnet3-info $dir/configs/ref.raw  > $dir/configs/temp.info 
  model_left_context=`fgrep 'left-context' $dir/configs/temp.info | awk '{print $2}'`
  model_right_context=`fgrep 'right-context' $dir/configs/temp.info | awk '{print $2}'`
  cat >$init_info <<EOF
frame_subsampling_factor $frame_subsampling_factor
langs $lang_list
model_left_context $model_left_context
model_right_context $model_right_context
EOF
  rm $dir/configs/temp.info
fi

model_left_context=$(awk '/^model_left_context/ {print $2;}' $dir/init/info.txt)
model_right_context=$(awk '/^model_right_context/ {print $2;}' $dir/init/info.txt)
if [ -z $model_left_context ]; then
    echo "ERROR: Cannot find entry for model_left_context in $dir/init/info.txt"
fi
if [ -z $model_right_context ]; then
    echo "ERROR: Cannot find entry for model_right_context in $dir/init/info.txt"
fi
egs_left_context=$[model_left_context+(frame_subsampling_factor/2)+extra_left_context]
egs_right_context=$[model_right_context+(frame_subsampling_factor/2)+extra_right_context]

if [ $stage -le 13 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
      lang_name=${lang_list[$lang_index]}
      tree_dir=${multi_ali_treedirs[$lang_index]}
      ali_dir=${multi_ali_dirs[$lang_index]}
      gmm_dir=${multi_gmm_dir[$lang_index]}

      cp $tree_dir/tree $dir/${lang_name}.tree
      echo "$0: creating phone language-model for $lang_name"
      $train_cmd $dir/den_fsts/log/make_phone_lm_${lang_name}.log \
        chain-est-phone-lm --num-extra-lm-states=2000 \
           "ark:gunzip -c $ali_dir/ali.*.gz | ali-to-phones $gmm_dir/final.mdl ark:- ark:- |" \
           $dir/den_fsts/${lang_name}.phone_lm.fst || exit 1
      echo "$0: creating denominator FST for $lang_name"
      copy-transition-model $tree_dir/final.mdl $dir/init/${lang_name}_trans.mdl  || exit 1 
      $train_cmd $dir/den_fsts/log/make_den_fst.log \
         chain-make-den-fst $dir/${lang_name}.tree \
            $dir/init/${lang_name}_trans.mdl $dir/den_fsts/${lang_name}.phone_lm.fst \
            $dir/den_fsts/${lang_name}.den.fst $dir/den_fsts/${lang_name}.normalization.fst || exit 1;
  done
fi

if [ $stage -le 14 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
      lang_name=${lang_list[$lang_index]}
      echo "$0: Generating raw egs for $lang_name"
      train_ivector_dir=${multi_ivector_dirs[$lang_index]}
      train_data_dir=${multi_data_dirs[$lang_index]}
      lat_dir=${multi_ali_latdirs[$lang_index]}
      if [ ! -f ${dir}/${lang_name}_processed_egs/.done ]; then
          steps/chain2/get_raw_egs.sh --cmd "$train_cmd" \
            --lang "$lang_name" \
            --online-ivector-dir $train_ivector_dir \
            --left-context $egs_left_context \
            --right-context $egs_right_context \
            --frame-subsampling-factor $frame_subsampling_factor \
            --alignment-subsampling-factor $frame_subsampling_factor \
            --frames-per-chunk $chunk_width \
            ${train_data_dir} ${dir} ${lat_dir} ${dir}/${lang_name}_raw_egs || exit 1

          echo "$0: Processing raw egs for $lang_name"
          steps/chain2/process_egs.sh  --cmd "$train_cmd" \
              ${dir}/${lang_name}_raw_egs ${dir}/${lang_name}_processed_egs || exit 1
          touch ${dir}/${lang_name}_processed_egs/.done
          rm -r ${dir}/${lang_name}_raw_egs # save space
      fi
  done
fi

if [ $stage -le 15 ]; then
    echo "$0: Combining egs"
    if [ ! -z "$lang2weight" ]; then
        egs_opts="--lang2weight '$lang2weight'"
    fi
    egs_dir_list=$(for lang_index in `seq 0 $[$num_langs-1]`;do lang_name=${lang_list[$lang_index]}; echo ${dir}/${lang_name}_processed_egs; done)
    
    steps/chain2/combine_egs.sh $egs_opts \
        --cmd "$train_cmd" \
        $num_langs $egs_dir_list ${dir}/egs
fi
[[ -z $common_egs_dir ]] && common_egs_dir=${dir}/egs

if [ $stage -le 16 ]; then
  [ ! -d ${dir}/egs/misc ] && mkdir  ${dir}/egs/misc
  echo "$0: Copying den.fst to ${dir}/egs/misc"
  for lang_index in `seq 0 $[$num_langs-1]`;do
      lang_name=${lang_list[$lang_index]}
      cp $dir/den_fsts/${lang_name}.*fst ${dir}/egs/misc/
      cp $dir/init/${lang_name}_trans.mdl ${dir}/egs/misc/${lang_name}.trans_mdl
      ln -rs $dir/egs/info.txt $dir/egs/info_${lang_name}.txt
  done
  echo "$0: Create a dummy transition model that is never used"
  first_lang_name=${lang_list[0]}
  [[ ! -f $dir/init/default_trans.mdl ]] && ln -r -s $dir/init/${first_lang_name}_trans.mdl $dir/init/default_trans.mdl
fi

if [ $stage -le 17 ]; then
    echo "$0: Preparing initial acoustic model"
    $cuda_cmd ${dir}/log/init_model.log \
           nnet3-init --srand=${srand} ${dir}/configs/final.config ${dir}/init/multi.raw || exit 1
fi

if [ $stage -le 18 ]; then
  echo "$0: Starting model training"
  steps/chain2/train.sh \
    --stage $train_stage --cmd "$cuda_cmd" \
    --multilingual-eg true \
    --xent-regularize $xent_regularize --leaky-hmm-coefficient 0.25  \
    --initial-effective-lrate $initial_effective_lrate \
    --final-effective-lrate $final_effective_lrate \
    --max-param-change $max_param_change \
    --groups-per-minibatch 128 \
    --srand 1 \
    --shuffle-buffer-size 5000 \
    --l2-regularize 5e-5 \
    --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
     $common_egs_dir $dir
fi

if [ $stage -le 19 ]; then
    echo "$0: Splitting models"
    frame_subsampling_factor=`fgrep "frame_subsampling_factor" $dir/init/info.txt | awk '{print $2}'`
    for lang_index in `seq 0 $[$num_langs-1]`;do
        lang_name=${lang_list[$lang_index]}
       [[ ! -d $dir/${lang_name} ]] && mkdir $dir/${lang_name}
       nnet3-copy --edits="rename-node old-name=output new-name=output-dummy; rename-node old-name=output-${lang_name} new-name=output" \
           $dir/final.raw - | \
           nnet3-am-init $dir/init/${lang_name}_trans.mdl - $dir/${lang_name}/final.mdl
       [[ ! -d $dir/${lang_name}/init ]] && mkdir $dir/${lang_name}/init
       params="frame_subsampling_factor model_left_context model_right_context feat_dim left_context left_context_initial right_context right_context_final ivector_dim frames_per_chunk"
       for param_name in $params; do
            grep -m 1 "^$param_name " $dir/init/info.txt
       done > $dir/${lang_name}/init/info.txt
    done
fi
