#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -u
set -o pipefail

. path.sh

# The following are the main parameters to modify
data_dir=data/train_si284   # Expecting whole data directory.
vad_dir=   # Output of prepare_unsad_data.sh. 
           # If provided, the speech labels and deriv weights will be 
           # copied into the output data directory.

num_data_reps=5   # Number of corrupted versions
foreground_snrs="20:10:15:5:0:-5"
background_snrs="20:10:15:5:2:0:-2:-5"

stage=0

# Parallel options
nj=4
cmd=run.pl

# Options for feature extraction
mfcc_config=conf/mfcc_hires_bp.conf
feat_suffix=hires_bp

# Data options
corrupt_only=false
speed_perturb=true
speeds="0.9 1.0 1.1"
resample_data_dir=false



. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

data_id=`basename ${data_dir}`

if [ ! -d RIRS_NOISES/ ]; then
  # Prepare MUSAN rirs and noises
  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  unzip rirs_noises.zip
fi

rvb_opts=()
# This is the config for the system using simulated RIRs and point-source noises
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
rvb_opts+=(--noise-set-parameters "0.1, RIRS_NOISES/pointsource_noises/background_noise_list")
rvb_opts+=(--noise-set-parameters "0.9, RIRS_NOISES/pointsource_noises/foreground_noise_list")

for f in RIRS_NOISES/simulated_rirs/smallroom/rir_list \
    RIRS_NOISES/simulated_rirs/mediumroom/rir_list \
    $data_dir/wav.scp; do 
  [ ! -f $f ] && echo "$0: Could not find $f" && exit 1
done

if $resample_data_dir; then
  sample_frequency=`cat $mfcc_config | perl -ne 'if (m/--sample-frequency=(\S+)/) { print $1; }'` 
  if [ -z "$sample_frequency" ]; then
    sample_frequency=16000
  fi

  utils/data/resample_data_dir.sh $sample_frequency ${data_dir} || exit 1
  data_id=`basename ${data_dir}`
  rvb_opts+=(--source-sampling-rate=$sample_frequency)
fi

corrupted_data_id=${data_id}_corrupted
clean_data_id=${data_id}_clean
noise_data_id=${data_id}_noise

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
    --max-noises-per-minute=2 \
    --output-additive-noise-dir=data/${noise_data_id} \
    --output-reverb-dir=data/${clean_data_id} \
    data/${data_id} data/${corrupted_data_id}
fi

corrupted_data_dir=data/${corrupted_data_id}
clean_data_dir=data/${clean_data_id}
noise_data_dir=data/${noise_data_id}

if $speed_perturb; then
  if [ $stage -le 2 ]; then
    ## Assuming whole data directories
    for x in $corrupted_data_dir $clean_data_dir $noise_data_dir; do
      cp $x/reco2dur $x/utt2dur
      utils/data/perturb_data_dir_speed_random.sh --speeds "$speeds" $x ${x}_spr
    done
  fi

  corrupted_data_dir=${corrupted_data_dir}_spr
  clean_data_dir=${clean_data_dir}_spr
  noise_data_dir=${noise_data_dir}_spr
  corrupted_data_id=${corrupted_data_id}_spr
  clean_data_id=${clean_data_id}_spr
  noise_data_id=${noise_data_id}_spr

  if [ $stage -le 3 ]; then
    utils/data/perturb_data_dir_volume.sh --scale-low 0.03125 --scale-high 2 \
      ${corrupted_data_dir}
    utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${clean_data_dir}
    utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${noise_data_dir}
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
    --cmd "$cmd" --nj $nj --write-utt2num-frames true \
    $corrupted_data_dir exp/make_${feat_suffix}/${corrupted_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $corrupted_data_dir exp/make_${feat_suffix}/${corrupted_data_id} $mfccdir
else
  corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
fi

if [ $stage -le 5 ]; then
  utils/copy_data_dir.sh $clean_data_dir ${clean_data_dir}_$feat_suffix
  clean_data_dir=${clean_data_dir}_$feat_suffix
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$cmd" --nj $nj \
    $clean_data_dir exp/make_${feat_suffix}/${clean_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $clean_data_dir exp/make_${feat_suffix}/${clean_data_id} $mfccdir
else
  clean_data_dir=${clean_data_dir}_$feat_suffix
fi 

if [ $stage -le 6 ]; then
  utils/copy_data_dir.sh $noise_data_dir ${noise_data_dir}_$feat_suffix
  noise_data_dir=${noise_data_dir}_$feat_suffix
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$cmd" --nj $nj \
    $noise_data_dir exp/make_${feat_suffix}/${noise_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $noise_data_dir exp/make_${feat_suffix}/${noise_data_id} $mfccdir
else
  noise_data_dir=${noise_data_dir}_$feat_suffix
fi 

targets_dir=irm_targets
if [ $stage -le 7 ]; then
  mkdir -p exp/make_log_snr/${corrupted_data_id}

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$targets_dir/storage $targets_dir/storage
  fi

  idct_params=`cat $mfcc_config | perl -e '
  $num_mel_bins = 23; $num_ceps = 13; $cepstral_lifter = 22.0; 
  while (<>) {
    chomp; 
    s/#.+//g; 
    if (m/^\s*$/) { next; }
    if (m/--num-mel-bins=(\S+)/) { 
      $num_mel_bins = $1; 
    } elsif (m/--num-ceps=(\S+)/) { 
      $num_ceps = $1; 
    } elsif (m/--cepstral-lifter=(\S+)/) { 
      $cepstral_lifter = $1; 
    }
  } 
  print "$num_mel_bins $num_ceps $cepstral_lifter";'`

  num_filters=`echo $idct_params | awk '{print $1}'`
  num_ceps=`echo $idct_params | awk '{print $2}'`
  cepstral_lifter=`echo $idct_params | awk '{print $3}'`
  echo "$num_filters $num_ceps $cepstral_lifter"

  mkdir -p exp/make_irm_targets/$corrupted_data_id
  utils/data/get_dct_matrix.py --get-idct-matrix=true \
    --num-filters=$num_filters --num-ceps=$num_ceps \
    --cepstral-lifter=$cepstral_lifter \
    exp/make_irm_targets/$corrupted_data_id/idct_matrix

  # Get log-IRM targets 
  steps/segmentation/make_snr_targets.sh \
    --nj $nj --cmd "$cmd" \
    --target-type Irm --compress false \
    --transform-matrix exp/make_irm_targets/$corrupted_data_id/idct_matrix \
    ${clean_data_dir} ${noise_data_dir} ${corrupted_data_dir} \
    exp/make_irm_targets/${corrupted_data_id} $targets_dir
fi


if [ $stage -le 8 ]; then
  if [ ! -z "$vad_dir" ]; then
    if [ ! -f $vad_dir/speech_labels.scp ]; then
      echo "$0: Could not find file $vad_dir/speech_labels.scp"
      exit 1
    fi
    
    cat $vad_dir/speech_labels.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
      sort -k1,1 > ${corrupted_data_dir}/speech_labels.scp
  
    if [ -f $vad_dir/deriv_weights.scp ]; then
      cat $vad_dir/deriv_weights.scp | \
        steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
        sort -k1,1 > ${corrupted_data_dir}/deriv_weights.scp
    fi
    
    if [ -f $vad_dir/deriv_weights_manual_seg.scp ]; then
      cat $vad_dir/deriv_weights_manual_seg.scp | \
        steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
        sort -k1,1 > ${corrupted_data_dir}/deriv_weights_for_irm_targets.scp
    fi
  fi
fi

exit 0
