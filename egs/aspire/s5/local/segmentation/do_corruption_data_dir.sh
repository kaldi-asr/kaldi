#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -u
set -o pipefail

. path.sh

stage=0
corruption_stage=-10
corrupt_only=false

# Data options
data_dir=data/train_si284   # Expecting whole data directory.
speed_perturb=true
num_data_reps=5   # Number of corrupted versions
snrs="20:10:15:5:0:-5"
foreground_snrs="20:10:15:5:0:-5"
background_snrs="20:10:15:5:0:-5"
base_rirs=simulated

# Parallel options
reco_nj=40  
cmd=queue.pl

# Options for feature extraction
mfcc_config=conf/mfcc_hires_bp.conf
feat_suffix=hires_bp

reco_vad_dir=   # Output of prepare_unsad_data.sh. 
                # If provided, the speech labels and deriv weights will be 
                # copied into the output data directory.

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

data_id=`basename ${data_dir}`

rvb_opts=()
if [ "$base_rirs" == "simulated" ]; then
  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)
else
  # This is the config for the JHU ASpIRE submission system
  rvb_opts+=(--rir-set-parameters "1.0, RIRS_NOISES/real_rirs_isotropic_noises/rir_list")
  rvb_opts+=(--noise-set-parameters RIRS_NOISES/real_rirs_isotropic_noises/noise_list)
fi

corrupted_data_id=${data_id}_corrupted

if [ $stage -le 1 ]; then
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix="rev" \
    --foreground-snrs=$foreground_snrs \
    --background-snrs=$background_snrs \
    --speech-rvb-probability=1 \
    --pointsource-noise-addition-probability=1 \
    --isotropic-noise-addition-probability=1 \
    --num-replications=$num_data_reps \
    --max-noises-per-minute=1 \
    data/${data_id} data/${corrupted_data_id}
fi

corrupted_data_dir=data/${corrupted_data_id}

if $speed_perturb; then
  if [ $stage -le 2 ]; then
    ## Assuming whole data directories
    for x in $corrupted_data_dir; do
      cp $x/reco2dur $x/utt2dur
      utils/data/perturb_data_dir_speed_random.sh $x ${x}_spr
    done
  fi

  corrupted_data_dir=${corrupted_data_dir}_spr
  corrupted_data_id=${corrupted_data_id}_spr

  if [ $stage -le 3 ]; then
    utils/data/perturb_data_dir_volume.sh --scale-low 0.03125 --scale-high 2 \
      ${corrupted_data_dir}
  fi
fi

if $corrupt_only; then
  echo "$0: Got corrupted data directory in ${corrupted_data_dir}"
  exit 0
fi

mfccdir=`basename $mfcc_config`
mfccdir=${mfccdir%%.conf}

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
fi

if [ $stage -le 4 ]; then
  utils/copy_data_dir.sh $corrupted_data_dir ${corrupted_data_dir}_$feat_suffix
  corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$cmd" --nj $reco_nj \
    $corrupted_data_dir exp/make_${feat_suffix}/${corrupted_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $corrupted_data_dir exp/make_${feat_suffix}/${corrupted_data_id} $mfccdir
else
  corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
fi 

if [ $stage -le 8 ]; then
  if [ ! -z "$reco_vad_dir" ]; then
    if [ ! -f $reco_vad_dir/speech_labels.scp ]; then
      echo "$0: Could not find file $reco_vad_dir/speech_labels.scp"
      exit 1
    fi
    
    cat $reco_vad_dir/speech_labels.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
      sort -k1,1 > ${corrupted_data_dir}/speech_labels.scp
  
    cat $reco_vad_dir/deriv_weights.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
      sort -k1,1 > ${corrupted_data_dir}/deriv_weights.scp
  fi
fi

exit 0
