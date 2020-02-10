#!/usr/bin/env bash

# Copyright 2016 Pegah Ghahremani

# This script is used for decoding multilingual model and it is called in
# local/nnet3/run_tdnn_multilingual.sh script.
# This script needs decoding data dir, which is prepared using
# eg/babel/s5d scripts (i.e. run-4-anydecode.sh).
# If --use-pitch is true, pitch feature is added to high-resolution MFCC features.
# If --use-bnf option is true, the --bnf-nnet-dir option, nnet3 model for
# bottleneck feature extraction, should be provided.

set -e
set -o pipefail


dir=dev10h.pem
kind=
use_pitch=true
use_pitch_ivector=false # If true, pitch feature is used in ivector extraction.
use_ivector=false
decode_stage=-1
nnet3_affix=
feat_suffix=
ivector_suffix=
iter=final
nj=30

# params for extracting bn features
use_bnf=false # If true, bottleneck feature is extracted and appended to input
              # for nnet3 model.
bnf_nnet_dir=exp/nnet3/multi_bnf_sp # dir for bottlneck nnet3 model
                                    # (used for bottleneck feature extraction)
use_ivector_bnf=false # If true, ivector used in extracting bottleneck features.

. conf/common_vars.sh || exit 1;

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $(basename $0) --dir <dir-type> <lang> <multilingual-nnet3-dir>"
  echo " e.g.: $(basename $0) --dir dev2h.pem ASM exp/nnet3/tdnn_multi_sp"
  exit 1
fi

lang=$1
nnet3_dir=$2

langconf=conf/$lang/lang.conf

if [ ! -f $langconf ]; then
  echo "$0: Language configuration $langconf does not exist! Use the "
  echo "configurations in ../../babel/s5d/conf/lang/$lang-* as a startup." && exit 1
fi
. $langconf || exit 1;
[ -f local.conf ] && . local.conf;

mfcc=mfcc/$lang
data=data/$lang
vector_suffix=_gb

dataset_dir=$data/$dir
dataset_id=$dir
dataset_type=${dir%%.*}

#By default, we want the script to accept how the dataset should be handled,
#i.e. of  what kind is the dataset
if [ -z ${kind} ] ; then
  if [ "$dataset_type" == "dev2h" ] || [ "$dataset_type" == "dev10h" ]; then
    dataset_kind=supervised
  else
    dataset_kind=unsupervised
  fi
else
  dataset_kind=$kind
fi

dataset=$(basename $dataset_dir)
mfccdir=mfcc_hires/$lang
mfcc_affix=""
hires_config="--mfcc-config conf/mfcc_hires.conf"
nnet3_data_dir=${dataset_dir}_hires
feat_suffix=_hires
ivec_feat_suffix=_hires
log_dir=exp/$lang/make_hires/$dataset

if $use_pitch_ivector; then
  ivec_feat_suffix=_hires_pitch
fi

if $use_pitch; then
  mfcc_affix="_pitch_online"
  hires_config="$hires_config --online-pitch-config conf/pitch.conf"
  mfccdir=mfcc_hires_pitch/lang
  nnet3_data_dir=${dataset_dir}_hires_pitch
  feat_suffix="_hires_pitch"
  log_dir=exp/$lang/make_hires_pitch/$dataset
fi


####################################################################
##
##  Feature extraction for decoding
##
####################################################################
echo ---------------------------------------------------------------------
echo "Preparing ${dataset_kind} data files in ${dataset_dir} on" `date`
echo ---------------------------------------------------------------------
if [ ! -f  $dataset_dir/.done ] ; then
  if [ ! -f ${nnet3_data_dir}/.mfcc.done ]; then
    echo ---------------------------------------------------------------------
    echo "Preparing ${dataset_kind} MFCC features in  ${nnet3_data_dir} and corresponding "
    echo "iVectors in exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset}${feat_suffix}${ivector_suffix} on" `date`
    echo ---------------------------------------------------------------------
    if [ ! -d ${nnet3_data_dir} ]; then
      utils/copy_data_dir.sh $data/$dataset ${nnet3_data_dir}
    fi

    steps/make_mfcc${mfcc_affix}.sh --nj $nj $hires_config \
        --cmd "$train_cmd" ${nnet3_data_dir} $log_dir $mfccdir;
    steps/compute_cmvn_stats.sh ${nnet3_data_dir} $log_dir $mfccdir;
    utils/fix_data_dir.sh ${nnet3_data_dir};
    touch ${nnet3_data_dir}/.mfcc.done
  fi
  touch $dataset_dir/.done
fi

ivector_dir=exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset}${ivec_feat_suffix}${ivector_suffix}
if $use_ivector && [ ! -f $ivector_dir/.ivector.done ];then
  extractor=exp/multi/nnet3${nnet3_affix}/extractor
  ivec_feat_suffix=$feat_suffix
  if $use_pitch && ! $use_pitch_ivector; then
    ivec_feat_suffix=_hires
    featdir=${dataset_dir}${feat_suffix}
    mfcc_only_dim=`feat-to-dim scp:$featdir/feats.scp - | awk '{print $1-3}'`
    steps/select_feats.sh --cmd "$train_cmd" --nj $nj 0-$[$mfcc_only_dim-1] \
      $featdir ${dataset_dir}${ivec_feat_suffix} || exit 1;
    steps/compute_cmvn_stats.sh ${dataset_dir}${ivec_feat_suffix} || exit 1;
  fi

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${dataset_dir}${ivec_feat_suffix} $extractor $ivector_dir || exit 1;
  touch $ivector_dir/.ivector.done
fi

if $use_bnf; then
  multi_ivector_dir=exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset}${ivec_feat_suffix}${ivector_suffix}

  ivector_for_bnf_opt=
  if $use_ivector_bnf;then ivector_for_bnf_opt="--ivector-dir $multi_ivector_dir"; fi

  bnf_data_dir=${dataset_dir}_bnf/$lang
  if [ ! -f $bnf_data_dir/.done ]; then
    steps/nnet3/make_bottleneck_features.sh --use-gpu true --nj 100 --cmd "$train_cmd" \
      $ivector_for_bnf_opt tdnn_bn.renorm \
      ${dataset_dir}${feat_suffix} $bnf_data_dir \
      $bnf_nnet_dir bnf/$lang exp/$lang/make_${dataset}_bnf || exit 1;
    touch $bnf_data_dir/.done
  else
    echo "$0: Skip Bottleneck feature extraction; You can force to run this step deleting $bnf_data_dir/.done."
  fi

  appended_bnf=${dataset_dir}${feat_suffix}_bnf
  if [ ! -f $appended_bnf/.done ]; then
    steps/append_feats.sh  --nj 16 --cmd "$train_cmd" \
      $bnf_data_dir ${dataset_dir}${feat_suffix} \
      ${dataset_dir}${feat_suffix}_bnf exp/$lang/append${feat_suffix}_bnf \
      mfcc${feat_suffix}_bnf/$lang || exit 1;

    steps/compute_cmvn_stats.sh $appended_bnf exp/$lang/make_cmvn${feat_suffix}_bnf \
      mfcc${feat_suffix}_bnf/$lang || exit 1;
    touch $appended_bnf/.done
  fi
  feat_suffix=${feat_suffix}_bnf
fi

####################################################################
##
## nnet3 model decoding
##
####################################################################
if [ ! -f exp/$lang/tri5/graph/HCLG.fst ];then
  utils/mkgraph.sh \
    data/$lang/lang exp/$lang/tri5 exp/$lang/tri5/graph |tee exp/$lang/tri5/mkgraph.log
fi

if [ -f $nnet3_dir/$lang/final.mdl ]; then
  decode=$nnet3_dir/$lang/decode_${dataset_id}
  feat_suffix=_hires
  ivec_feat_suffix=_hires

  # suffix for using other features such as pitch
  if $use_pitch; then
    feat_suffix=${feat_suffix}_pitch
  fi
  if $use_pitch_ivector; then
    ivec_feat_suffix=_hires_pitch
  fi
  if $use_bnf; then
    feat_suffix=${feat_suffix}_bnf
  fi
  ivector_opts=
  if $use_ivector; then
    ivector_opts="--online-ivector-dir exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset_id}${ivec_feat_suffix}${ivector_suffix}"
  fi
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    score_opts="--skip-scoring false"
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" $iter_opt \
          --stage $decode_stage \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          $score_opts $ivector_opts \
          exp/$lang/tri5/graph ${dataset_dir}${feat_suffix} $decode | tee $decode/decode.log

    touch $decode/.done
  fi
fi

echo "Everything looking good...."
exit 0
