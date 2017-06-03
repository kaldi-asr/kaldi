#!/bin/bash
set -e
set -o pipefail


dir=dev10h.pem
kind=
data_only=false
skip_kws=false
skip_scoring=
extra_kws=true
vocab_kws=false
tri5_only=false
use_pitch=true
use_pitch_ivector=false # if true, pitch feature used in ivector extraction.
use_ivector=false
use_bnf=false
pitch_conf=conf/pitch.conf
wip=0.5
decode_stage=-1
nnet3_affix=
nnet3_dir=nnet3/tdnn_sp
is_rnn=false
extra_left_context=0
extra_right_context=0
frames_per_chunk=0
feat_suffix=
ivector_suffix=
iter=final

# params for extracting bn features
multidir=exp/nnet3/multi_bnf_sp
dump_bnf_dir=bnf
bnf_layer=5


. conf/common_vars.sh || exit 1;

. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0) --dir <dir-type> <lang>"
  echo " e.g.: $(basename $0) --dir dev2h.pem ASM"
  exit 1
fi

lang=$1


langconf=conf/$lang/lang.conf

[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
. $langconf || exit 1;
[ -f local.conf ] && . local.conf;

mfcc=mfcc/$lang
plp=plp/$lang
data=data/$lang
vector_suffix=_gb
#This seems to be the only functioning way how to ensure the comple
#set of scripts will exit when sourcing several of them together
#Otherwise, the CTRL-C just terminates the deepest sourced script ?
# Let shell functions inherit ERR trap.  Same as `set -E'.
set -o errtrace
trap "echo Exited!; exit;" SIGINT SIGTERM

# Set proxy search parameters for the extended lexicon case.
if [ -f $data/.extlex ]; then
  proxy_phone_beam=$extlex_proxy_phone_beam
  proxy_phone_nbest=$extlex_proxy_phone_nbest
  proxy_beam=$extlex_proxy_beam
  proxy_nbest=$extlex_proxy_nbest
fi

dataset_segments=${dir##*.}
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
data_dir=${dataset_dir}_hires
feat_suffix=_hires
ivec_feat_suffix=_hires
log_dir=exp/$lang/make_hires/$dataset

if $use_pitch_ivector; then
  ivec_feat_suffix=_hires_pitch
fi

if $use_pitch; then
  mfcc_affix="_pitch_online"
  hires_config="$hires_config --online-pitch-config $pitch_conf"
  mfccdir=mfcc_hires_pitch/lang
  data_dir=${dataset_dir}_hires_pitch
  feat_suffix="_hires_pitch"
  log_dir=exp/$lang/make_hires_pitch/$dataset
fi

if [ -z $dataset_segments ]; then
  echo "You have to specify the segmentation type as well"
  echo "If you are trying to decode the PEM segmentation dir"
  echo "such as data/dev10h, specify dev10h.pem"
  echo "The valid segmentations types are:"
  echo "\tpem   #PEM segmentation"
  echo "\tuem   #UEM segmentation in the CMU database format"
  echo "\tseg   #UEM segmentation (kaldi-native)"
fi

if [ -z "${skip_scoring}" ] ; then
  if [ "$dataset_kind" == "unsupervised" ]; then
    skip_scoring=true
  else
    skip_scoring=false
  fi
fi

#The $dataset_type value will be the dataset name without any extrension
eval my_data_dir=( "\${${dataset_type}_data_dir[@]}" )
eval my_data_list=( "\${${dataset_type}_data_list[@]}" )
if [ -z $my_data_dir ] || [ -z $my_data_list ] ; then
  echo "Error: The dir you specified ($dataset_id) does not have existing config";
  exit 1
fi

eval my_stm_file=\$${dataset_type}_stm_file
eval my_ecf_file=\$${dataset_type}_ecf_file
eval my_rttm_file=\$${dataset_type}_rttm_file
eval my_nj=\$${dataset_type}_nj  #for shadow, this will be re-set when appropriate

if [ -z "$my_nj" ]; then
  echo >&2 "You didn't specify the number of jobs -- variable \"${dataset_type}_nj\" not defined."
  exit 1
fi

my_subset_ecf=false
eval ind=\${${dataset_type}_subset_ecf+x}
if [ "$ind" == "x" ] ; then
  eval my_subset_ecf=\$${dataset_type}_subset_ecf
fi

declare -A my_kwlists=()
eval my_kwlist_keys="\${!${dataset_type}_kwlists[@]}"
for key in $my_kwlist_keys  # make sure you include the quotes there
do
  eval my_kwlist_val="\${${dataset_type}_kwlists[$key]}"
  my_kwlists["$key"]="${my_kwlist_val}"
done

#Just a minor safety precaution to prevent using incorrect settings
#The dataset_* variables should be used.
set -e
set -o pipefail
set -u
unset dir
unset kind

function make_plp {
  target=$1
  logdir=$2
  output=$3
  if $use_pitch; then
    steps/make_plp_pitch.sh --cmd "$decode_cmd" --nj $my_nj $target $logdir $output
  else
    steps/make_plp.sh --cmd "$decode_cmd" --nj $my_nj $target $logdir $output
  fi
  utils/fix_data_dir.sh $target
  steps/compute_cmvn_stats.sh $target $logdir $output
  utils/fix_data_dir.sh $target
}

function check_variables_are_set {
  for variable in $mandatory_variables ; do
    if ! declare -p $variable ; then
      echo "Mandatory variable ${variable/my/$dataset_type} is not set! "
      echo "You should probably set the variable in the config file "
      exit 1
    else
      declare -p $variable
    fi
  done

  if [ ! -z ${optional_variables+x} ] ; then
    for variable in $optional_variables ; do
      eval my_variable=\$${variable}
      echo "$variable=$my_variable"
    done
  fi
}

if [ ! -f $data/raw_${dataset_type}_data/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the ${dataset_type} set"
  echo ---------------------------------------------------------------------

  l1=${#my_data_dir[*]}
  l2=${#my_data_list[*]}
  if [ "$l1" -ne "$l2" ]; then
    echo "Error, the number of source files lists is not the same as the number of source dirs!"
    exit 1
  fi

  resource_string=""
  if [ "$dataset_kind" == "unsupervised" ]; then
    resource_string+=" --ignore-missing-txt true"
  fi

  for i in `seq 0 $(($l1 - 1))`; do
    resource_string+=" ${my_data_dir[$i]} "
    resource_string+=" ${my_data_list[$i]} "
  done
  local/make_corpus_subset.sh $resource_string ./$data/raw_${dataset_type}_data
  touch $data/raw_${dataset_type}_data/.done
fi
my_data_dir=`readlink -f ./$data/raw_${dataset_type}_data`
[ -f $my_data_dir/filelist.list ] && my_data_list=$my_data_dir/filelist.list
nj_max=`cat $my_data_list | wc -l` || nj_max=`ls $my_data_dir/audio | wc -l`

if [ "$nj_max" -lt "$my_nj" ] ; then
  echo "Number of jobs ($my_nj) is too big!"
  echo "The maximum reasonable number of jobs is $nj_max"
  my_nj=$nj_max
fi

#####################################################################
#
# Audio data directory preparation
#
#####################################################################
echo ---------------------------------------------------------------------
echo "Preparing ${dataset_kind} data files in ${dataset_dir} on" `date`
echo ---------------------------------------------------------------------
if [ ! -f  $dataset_dir/.done ] ; then
  if [ "$dataset_kind" == "supervised" ]; then
    if [ "$dataset_segments" == "seg" ]; then
      . ./local/datasets/supervised_seg.sh || exit 1
    elif [ "$dataset_segments" == "uem" ]; then
      . ./local/datasets/supervised_uem.sh || exit 1
    elif [ "$dataset_segments" == "pem" ]; then
      . ./local/datasets/supervised_pem.sh || exit 1
    else
      echo "Unknown type of the dataset: \"$dataset_segments\"!";
      echo "Valid dataset types are: seg, uem, pem";
      exit 1
    fi
  elif [ "$dataset_kind" == "unsupervised" ] ; then
    if [ "$dataset_segments" == "seg" ] ; then
      . ./local/datasets/unsupervised_seg.sh
    elif [ "$dataset_segments" == "uem" ] ; then
      . ./local/datasets/unsupervised_uem.sh
    elif [ "$dataset_segments" == "pem" ] ; then
      ##This combination does not really makes sense,
      ##Because the PEM is that we get the segmentation
      ##and because of the format of the segment files
      ##the transcript as well
      echo "ERROR: $dataset_segments combined with $dataset_type"
      echo "does not really make any sense!"
      exit 1
      #. ./local/datasets/unsupervised_pem.sh
    else
      echo "Unknown type of the dataset: \"$dataset_segments\"!";
      echo "Valid dataset types are: seg, uem, pem";
      exit 1
    fi
  else
    echo "Unknown kind of the dataset: \"$dataset_kind\"!";
    echo "Valid dataset kinds are: supervised, unsupervised, shadow";
    exit 1
  fi

  if [ ! -f ${dataset_dir}/.plp.done ]; then
    echo ---------------------------------------------------------------------
    echo "Preparing ${dataset_kind} parametrization files in ${dataset_dir} on" `date`
    echo ---------------------------------------------------------------------
    make_plp ${dataset_dir} exp/$lang/make_plp/${dataset_id} plp/$lang
    touch ${dataset_dir}/.plp.done
  fi


  if [ ! -f ${data_dir}/.mfcc.done ]; then
    echo ---------------------------------------------------------------------
    echo "Preparing ${dataset_kind} MFCC features in  ${data_dir} and corresponding iVectors in exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset}${feat_suffix}${ivector_suffix} on" `date`
    echo ---------------------------------------------------------------------
    if [ ! -d ${data_dir} ]; then
      utils/copy_data_dir.sh $data/$dataset ${data_dir}
    fi


    steps/make_mfcc${mfcc_affix}.sh --nj $my_nj $hires_config \
        --cmd "$train_cmd" ${data_dir} $log_dir $mfccdir;
    steps/compute_cmvn_stats.sh ${data_dir} $log_dir $mfccdir;
    utils/fix_data_dir.sh ${data_dir};
    touch ${data_dir}/.mfcc.done
  fi
  touch $dataset_dir/.done
fi

# extract ivector
dataset=$(basename $dataset_dir)
ivector_dir=exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset}${ivec_feat_suffix}${ivector_suffix}
if $use_ivector && [ ! -f $ivector_dir/.ivector.done ];then
  extractor=exp/multi/nnet3${nnet3_affix}/extractor
  ivec_feat_suffix=$feat_suffix
  if $use_pitch && ! $use_pitch_ivector; then
    ivec_feat_suffix=_hires
    featdir=${dataset_dir}${feat_suffix}
    mfcc_only_dim=`feat-to-dim scp:$featdir/feats.scp - | awk '{print $1-3}'`
    steps/select_feats.sh --cmd "$train_cmd" --nj $my_nj 0-$[$mfcc_only_dim-1] \
      $featdir ${dataset_dir}${ivec_feat_suffix} || exit 1;
  fi

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $my_nj \
    ${dataset_dir}${ivec_feat_suffix} $extractor $ivector_dir || exit 1;
  touch $ivector_dir/.ivector.done
fi

if $use_bnf; then
  # put the archives in ${dump_bnf_dir}/.
  dataset=$(basename $dataset_dir)
  multi_ivector_dir=exp/$lang/nnet3${nnet3_affix}/ivectors_${dataset}${ivec_feat_suffix}${ivector_suffix}
  bnf_data_dir=${dataset_dir}_bnf/$lang
  if [ ! -f $bnf_data_dir/.done ]; then
  steps/nnet3/dump_bottleneck_features.sh --use-gpu true --nj 100 --cmd "$train_cmd" \
    --ivector-dir $multi_ivector_dir \
    --feat-type raw \
    ${dataset_dir}${feat_suffix} $bnf_data_dir \
    $multidir $dump_bnf_dir/$lang exp/$lang/make_${dataset}_bnf || exit 1;
  touch $bnf_data_dir/.done
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

#####################################################################
#
# KWS data directory preparation
#
#####################################################################
echo ---------------------------------------------------------------------
echo "Preparing kws data files in ${dataset_dir} on" `date`
echo ---------------------------------------------------------------------

if ! $skip_kws ; then
  if  $extra_kws ; then
    L1_lex=data/local/lexiconp.txt
    . ./local/datasets/extra_kws.sh || exit 1
  fi
  if  $vocab_kws ; then
    . ./local/datasets/vocab_kws.sh || exit 1
  fi
fi
if $data_only ; then
  echo "Exiting, as data-only was requested..."
  exit 0;
fi

####################################################################
## FMLLR decoding
##
####################################################################
decode=exp/$lang/tri5/decode_${dataset_id}
if [ ! -f exp/$lang/tri5/graph/HCLG.fst ];then
  utils/mkgraph.sh \
    data/$lang/lang exp/$lang/tri5 exp/$lang/tri5/graph |tee exp/$lang/tri5/mkgraph.log
fi
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/$lang/lang exp/$lang/tri5 exp/$lang/tri5/graph |tee exp/$lang/tri5/mkgraph.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4\
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    exp/$lang/tri5/graph ${dataset_dir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi


if $tri5_only; then
  echo "--tri5-only is true. So exiting."
  exit 0
fi

####################################################################
##
## nnet3 model decoding
##
####################################################################

if [ -f $nnet3_dir/$lang/final.mdl ]; then
  decode=$nnet3_dir/$lang/decode_${dataset_id}
  rnn_opts=
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
  if [ "$is_rnn" == "true" ]; then
    rnn_opts=" --extra-left-context $extra_left_context --extra-right-context $extra_right_context  --frames-per-chunk $frames_per_chunk "
  fi
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    score_opts="--skip-scoring false"
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/nnet3/decode.sh --nj $my_nj --cmd "$decode_cmd" $iter_opt $rnn_opts \
          --stage $decode_stage \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          $score_opts $ivector_opts \
          exp/$lang/tri5/graph ${dataset_dir}${feat_suffix} $decode | tee $decode/decode.log

    touch $decode/.done
  fi
fi

echo "Everything looking good...."
exit 0
