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
data_dir=data/train_si284   # Excpecting non-whole data directory
num_data_reps=5   # Number of corrupted versions
snrs="20:10:15:5:0:-5"
foreground_snrs="20:10:15:5:0:-5"
background_snrs="20:10:15:5:0:-5"
overlap_snrs="5:2:1:0:-1:-2"
overlap_labels_dir=overlap_labels

# Parallel options
nj=40
cmd=queue.pl

# Options for feature extraction
mfcc_config=conf/mfcc_hires_bp.conf
feat_suffix=hires_bp
energy_config=conf/log_energy.conf

utt_vad_dir=

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

rvb_opts=()
# This is the config for the system using simulated RIRs and point-source noises
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
rvb_opts+=(--speech-segments-set-parameters="$data_dir/wav.scp,$data_dir/segments")

if [ $stage -le 0 ]; then
  steps/segmentation/get_data_dir_with_segmented_wav.py \
    $data_dir ${data_dir}_seg
fi

data_dir=${data_dir}_seg

data_id=`basename ${data_dir}`

corrupted_data_id=${data_id}_ovlp_corrupted
clean_data_id=${data_id}_ovlp_clean
noise_data_id=${data_id}_ovlp_noise

utils/data/get_reco2dur.sh --cmd $cmd --nj 40 $data_dir

if [ $stage -le 1 ]; then
  python steps/data/make_corrupted_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix="ovlp" \
    --overlap-snrs=$overlap_snrs \
    --speech-rvb-probability=1 \
    --overlapping-speech-addition-probability=1 \
    --num-replications=$num_data_reps \
    --min-overlapping-segments-per-minute=1 \
    --max-overlapping-segments-per-minute=1 \
    --output-additive-noise-dir=data/${noise_data_id} \
    --output-reverb-dir=data/${clean_data_id} \
    ${data_dir} data/${corrupted_data_id}
fi

clean_data_dir=data/${clean_data_id}
corrupted_data_dir=data/${corrupted_data_id}
noise_data_dir=data/${noise_data_id}
orig_corrupted_data_dir=data/${corrupted_data_id}

if false; then
  if [ $stage -le 2 ]; then
    for x in $clean_data_dir $corrupted_data_dir $noise_data_dir; do
      utils/data/perturb_data_dir_speed_3way.sh $x ${x}_sp
    done
  fi

  corrupted_data_dir=${corrupted_data_dir}_sp
  clean_data_dir=${clean_data_dir}_sp
  noise_data_dir=${noise_data_dir}_sp

  corrupted_data_id=${corrupted_data_id}_sp
  clean_data_id=${clean_data_id}_sp
  noise_data_id=${noise_data_id}_sp
fi

if [ $stage -le 3 ]; then
  utils/data/perturb_data_dir_volume.sh --scale-low 0.03125 --scale-high 2 ${corrupted_data_dir}
  utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${clean_data_dir}
  utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${noise_data_dir}
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
    --cmd "$cmd" --nj $nj \
    $corrupted_data_dir exp/make_${feat_suffix}/${corrupted_data_id} $mfccdir
else
  corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
fi

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d log_energy/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/log_energy/storage log_energy/storage
fi

if [ $stage -le 5 ]; then
  utils/copy_data_dir.sh $clean_data_dir ${clean_data_dir}_log_energy
  steps/make_mfcc.sh --mfcc-config conf/log_energy.conf \
    --cmd "$cmd" --nj $nj ${clean_data_dir}_log_energy \
    exp/make_log_energy/${clean_data_id} log_energy
fi

if [ $stage -le 6 ]; then
  utils/copy_data_dir.sh $noise_data_dir ${noise_data_dir}_log_energy
  steps/make_mfcc.sh --mfcc-config conf/log_energy.conf \
    --cmd "$cmd" --nj $nj ${noise_data_dir}_log_energy \
    exp/make_log_energy/${noise_data_id} log_energy
fi

targets_dir=log_snr
if [ $stage -le 7 ]; then
  mkdir -p exp/make_log_snr/${corrupted_data_id}

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$targets_dir/storage $targets_dir/storage
  fi

  # Get log-SNR targets 
  steps/segmentation/make_snr_targets.sh \
    --nj $nj --cmd "$cmd" \
    --target-type Snr --compress false \
    ${clean_data_dir}_log_energy ${noise_data_dir}_log_energy ${corrupted_data_dir} \
    exp/make_log_snr/${corrupted_data_id} $targets_dir
fi

exit 0
  
if [ $stage -le 5 ]; then
  # clean here is the reverberated first-speaker signal
  utils/copy_data_dir.sh $clean_data_dir ${clean_data_dir}_$feat_suffix
  clean_data_dir=${clean_data_dir}_$feat_suffix
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$cmd" --nj $nj \
    $clean_data_dir exp/make_${feat_suffix}/${clean_data_id} $mfccdir
else
  clean_data_dir=${clean_data_dir}_$feat_suffix
fi

if [ $stage -le 6 ]; then
  # noise here is the reverberated second-speaker signal
  utils/copy_data_dir.sh $noise_data_dir ${noise_data_dir}_$feat_suffix
  noise_data_dir=${noise_data_dir}_$feat_suffix
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$cmd" --nj $nj \
    $noise_data_dir exp/make_${feat_suffix}/${noise_data_id} $mfccdir
else
  noise_data_dir=${noise_data_dir}_$feat_suffix
fi

targets_dir=irm_targets
if [ $stage -le 8 ]; then
  mkdir -p exp/make_irm_targets/${corrupted_data_id}

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$targets_dir/storage $targets_dir/storage
  fi

  # Get SNR targets only for the overlapped speech labels.
  steps/segmentation/make_snr_targets.sh \
    --nj $nj --cmd "$cmd --max-jobs-run $max_jobs_run" \
    --target-type Irm --compress false --apply-exp true \
    --ali-rspecifier "ark,s,cs:cat ${corrupted_data_dir}/sad_seg.scp | segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames scp:- ark:- |" \
    overlapped_speech_labels.scp \
    --silence-phones 0 \
    ${clean_data_dir} ${noise_data_dir} ${corrupted_data_dir} \
    exp/make_irm_targets/${corrupted_data_id} $targets_dir
fi

exit 0
