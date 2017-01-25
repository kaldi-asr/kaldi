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
speed_perturb=true
num_data_reps=5   # Number of corrupted versions
snrs="20:10:15:5:0:-5"
foreground_snrs="20:10:15:5:0:-5"
background_snrs="20:10:15:5:0:-5"
overlap_snrs="5:2:1:0:-1:-2"
# Whole-data directory corresponding to data_dir
whole_data_dir=data/train_si284_whole   
overlap_labels_dir=overlap_labels

# Parallel options
reco_nj=40
nj=40
cmd=queue.pl

# Options for feature extraction
mfcc_config=conf/mfcc_hires_bp.conf
feat_suffix=hires_bp
energy_config=conf/log_energy.conf

reco_vad_dir=   # Output of prepare_unsad_data.sh. 
                # If provided, the speech labels and deriv weights will be 
                # copied into the output data directory.
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

whole_data_id=`basename ${whole_data_dir}`

corrupted_data_id=${whole_data_id}_ovlp_corrupted
clean_data_id=${whole_data_id}_ovlp_clean
noise_data_id=${whole_data_id}_ovlp_noise

if [ $stage -le 1 ]; then
  python steps/data/make_corrupted_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix="ovlp" \
    --overlap-snrs=$overlap_snrs \
    --speech-rvb-probability=1 \
    --overlapping-speech-addition-probability=1 \
    --num-replications=$num_data_reps \
    --min-overlapping-segments-per-minute=5 \
    --max-overlapping-segments-per-minute=20 \
    --output-additive-noise-dir=data/${noise_data_id} \
    --output-reverb-dir=data/${clean_data_id} \
    data/${whole_data_id} data/${corrupted_data_id}
fi

if $dry_run; then
  exit 0
fi

clean_data_dir=data/${clean_data_id}
corrupted_data_dir=data/${corrupted_data_id}
noise_data_dir=data/${noise_data_id}
orig_corrupted_data_dir=$corrupted_data_dir

if $speed_perturb; then
  if [ $stage -le 2 ]; then
    ## Assuming whole data directories
    for x in $clean_data_dir $corrupted_data_dir $noise_data_dir; do
      cp $x/reco2dur $x/utt2dur
      utils/data/perturb_data_dir_speed_3way.sh $x ${x}_sp
    done
  fi

  corrupted_data_dir=${corrupted_data_dir}_sp
  clean_data_dir=${clean_data_dir}_sp
  noise_data_dir=${noise_data_dir}_sp

  corrupted_data_id=${corrupted_data_id}_sp
  clean_data_id=${clean_data_id}_sp
  noise_data_id=${noise_data_id}_sp

  if [ $stage -le 3 ]; then
    utils/data/perturb_data_dir_volume.sh --scale-low 0.03125 --scale-high 2 --force true ${corrupted_data_dir}
    utils/data/perturb_data_dir_volume.sh --force true --reco2vol ${corrupted_data_dir}/reco2vol ${clean_data_dir}
    utils/data/perturb_data_dir_volume.sh --force true --reco2vol ${corrupted_data_dir}/reco2vol ${noise_data_dir}
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
    --cmd "$train_cmd" --nj $reco_nj \
    $corrupted_data_dir exp/make_${feat_suffix}/${corrupted_data_id} $mfccdir
fi

if [ $stage -le 5 ]; then
  steps/make_mfcc.sh --mfcc-config $energy_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $clean_data_dir exp/make_log_energy/${clean_data_id} log_energy_feats
fi

if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --mfcc-config $energy_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $noise_data_dir exp/make_log_energy/${noise_data_id} log_energy_feats
fi

if [ -z "$reco_vad_dir" ]; then
  echo "reco-vad-dir must be provided"
  exit 1
fi

targets_dir=irm_targets
if [ $stage -le 8 ]; then
  mkdir -p exp/make_irm_targets/${corrupted_data_id}

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$targets_dir/storage $targets_dir/storage
  fi

  steps/segmentation/make_snr_targets.sh \
    --nj $nj --cmd "$train_cmd --max-jobs-run $max_jobs_run" \
    --target-type Irm --compress true --apply-exp false \
    ${clean_data_dir} ${noise_data_dir} ${corrupted_data_dir} \
    exp/make_irm_targets/${corrupted_data_id} $targets_dir
fi

# Combine the VAD from the base recording and the VAD from the overlapping segments
# to create per-frame labels of the number of overlapping speech segments
# Unreliable segments are regions where no VAD labels were available for the
# overlapping segments. These can be later removed by setting deriv weights to 0.

# Data dirs without speed perturbation
overlap_dir=exp/make_overlap_labels/${corrupted_data_id}
unreliable_dir=exp/make_overlap_labels/unreliable_${corrupted_data_id}
overlap_data_dir=$overlap_dir/overlap_data
unreliable_data_dir=$overlap_dir/unreliable_data

mkdir -p $unreliable_dir

if [ $stage -le 8 ]; then
  cat $reco_vad_dir/sad_seg.scp | \
    steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps "ovlp" \
    | sort -k1,1 > ${corrupted_data_dir}/sad_seg.scp
  utils/data/get_utt2num_frames.sh $corrupted_data_dir
  utils/split_data.sh --per-reco ${orig_corrupted_data_dir} $reco_nj

  $train_cmd JOB=1:$reco_nj $overlap_dir/log/get_overlap_seg.JOB.log \
    segmentation-init-from-overlap-info --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    "scp:utils/filter_scp.pl ${orig_corrupted_data_dir}/split${reco_nj}reco/JOB/utt2spk $corrupted_data_dir/sad_seg.scp |" \
    ark,t:$orig_corrupted_data_dir/overlapped_segments_info.txt \
    scp:$utt_vad_dir/sad_seg.scp ark:- ark:$unreliable_dir/unreliable_seg_speed_unperturbed.JOB.ark \| \
    segmentation-copy --keep-label=1 ark:- ark:- \| \
    segmentation-get-stats --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    ark:- ark:- ark:/dev/null \| \
    segmentation-init-from-ali ark:- ark:$overlap_dir/overlap_seg_speed_unperturbed.JOB.ark
fi

if [ $stage -le 9 ]; then
  mkdir -p $overlap_data_dir $unreliable_data_dir
  cp $orig_corrupted_data_dir/wav.scp $overlap_data_dir
  cp $orig_corrupted_data_dir/wav.scp $unreliable_data_dir

  # Create segments where there is definitely an overlap.
  # Assume no more than 10 speakers overlap.
  $train_cmd JOB=1:$reco_nj $overlap_dir/log/process_to_segments.JOB.log \
    segmentation-post-process --remove-labels=0:1 \
    ark:$overlap_dir/overlap_seg_speed_unperturbed.JOB.ark ark:- \| \
    segmentation-post-process --merge-labels=2:3:4:5:6:7:8:9:10 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-to-segments ark:- ark:$overlap_data_dir/utt2spk.JOB $overlap_data_dir/segments.JOB

  $train_cmd JOB=1:$reco_nj $overlap_dir/log/get_unreliable_segments.JOB.log \
    segmentation-to-segments --single-speaker \
    ark:$unreliable_dir/unreliable_seg_speed_unperturbed.JOB.ark \
    ark:$unreliable_data_dir/utt2spk.JOB $unreliable_data_dir/segments.JOB

  for n in `seq $reco_nj`; do cat $overlap_data_dir/utt2spk.$n; done > $overlap_data_dir/utt2spk
  for n in `seq $reco_nj`; do cat $overlap_data_dir/segments.$n; done > $overlap_data_dir/segments
  for n in `seq $reco_nj`; do cat $unreliable_data_dir/utt2spk.$n; done > $unreliable_data_dir/utt2spk
  for n in `seq $reco_nj`; do cat $unreliable_data_dir/segments.$n; done > $unreliable_data_dir/segments

  utils/fix_data_dir.sh $overlap_data_dir
  utils/fix_data_dir.sh $unreliable_data_dir

  if $speed_perturb; then
    utils/data/perturb_data_dir_speed_3way.sh $overlap_data_dir ${overlap_data_dir}_sp
    utils/data/perturb_data_dir_speed_3way.sh $unreliable_data_dir ${unreliable_data_dir}_sp
  fi
fi

if $speed_perturb; then
  overlap_data_dir=${overlap_data_dir}_sp
  unreliable_data_dir=${unreliable_data_dir}_sp
fi

# make $overlap_labels_dir an absolute pathname.
overlap_labels_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $overlap_labels_dir ${PWD}`

if [ $stage -le 10 ]; then
  utils/split_data.sh --per-reco ${overlap_data_dir} $reco_nj

  $train_cmd JOB=1:$reco_nj $overlap_dir/log/get_overlap_speech_labels.JOB.log \
    utils/data/get_reco2utt.sh ${overlap_data_dir}/split${reco_nj}reco/JOB '&&' \
    segmentation-init-from-segments --shift-to-zero=false \
    ${overlap_data_dir}/split${reco_nj}reco/JOB/segments ark:- \| \
    segmentation-combine-segments-to-recordings ark:- ark,t:${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt \
    ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- \
    ark,scp:$overlap_labels_dir/overlapped_speech_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/overlapped_speech_${corrupted_data_id}.JOB.scp
fi

for n in `seq $reco_nj`; do
  cat $overlap_labels_dir/overlapped_speech_${corrupted_data_id}.$n.scp
done > ${corrupted_data_dir}/overlapped_speech_labels.scp

if [ $stage -le 11 ]; then
  utils/data/get_reco2utt.sh ${unreliable_data_dir}

  # First convert the unreliable segments into a recording-level segmentation.
  # Initialize a segmentation from utt2num_frames and set to 0, the regions
  # of unreliable segments. At this stage deriv weights is 1 for all but the
  # unreliable segment regions.
  # Initialize a segmentation from the VAD labels and retain only the speech segments.
  # Intersect this with the deriv weights segmentation from above. At this stage
  # deriv weights is 1 for only the regions where base VAD label is 1 and
  # the overlapping segment is not unreliable. Convert this to deriv weights.
  $train_cmd JOB=1:$reco_nj $unreliable_dir/log/get_deriv_weights.JOB.log\
    segmentation-init-from-segments --shift-to-zero=false \
    "utils/filter_scp.pl -f 2 ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt ${unreliable_data_dir}/segments |" ark:- \| \
    segmentation-combine-segments-to-recordings ark:- "ark,t:utils/filter_scp.pl ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt ${unreliable_data_dir}/reco2utt |" \
    ark:- \| \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=0 --ignore-missing \
    "ark:utils/filter_scp.pl ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt $corrupted_data_dir/utt2num_frames | segmentation-init-from-lengths ark,t:- ark:- |" \
    ark:- ark:- \| \
    segmentation-intersect-segments --mismatch-label=0 \
    "ark:utils/filter_scp.pl ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt $corrupted_data_dir/sad_seg.scp | segmentation-post-process --remove-labels=0:2:3 scp:- ark:- |" \
    ark:- ark:- \| \
    segmentation-post-process --remove-labels=0 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
    ark,scp:$overlap_labels_dir/deriv_weights_for_overlapped_speech.JOB.ark,$overlap_labels_dir/deriv_weights_for_overlapped_speech.JOB.scp

  for n in `seq $reco_nj`; do
    cat $overlap_labels_dir/deriv_weights_for_overlapped_speech.${n}.scp
  done > $corrupted_data_dir/deriv_weights_for_overlapped_speech.scp
fi

exit 0
