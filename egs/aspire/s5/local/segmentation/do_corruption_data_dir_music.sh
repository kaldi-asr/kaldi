#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -u
set -o pipefail

. path.sh

# The following are the main parameters to modify
data_dir=data/train_si284
vad_dir=      # Location of directory with VAD labels

num_data_reps=5   # Number of corrupted versions
foreground_snrs="5:2:1:0:-2:-5:-10:-20"
background_snrs="5:2:1:0:-2:-5:-10:-20"

stage=0

# Parallel options
nj=4
cmd=run.pl


# Options for feature extraction
mfcc_config=conf/mfcc_hires_bp.conf
feat_suffix=hires_bp

corrupt_only=false
speed_perturb=true
speeds="0.9 1.0 1.1"
resample_data_dir=false

label_dir=music_labels    # Directory to dump music labels

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

if [ ! -d RIRS_NOISES/music ]; then
  wget --no-check-certificate http://www.openslr.org/resources/17/musan.tar.gz
  tar -xvf musan.tar.gz 
  
  # Prepare MUSAN music
  local/segmentation/prepare_musan_music.sh musan RIRS_NOISES/music
fi

rvb_opts=()
# This is the config for the system using simulated RIRs and point-source noises
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
rvb_opts+=(--noise-set-parameters RIRS_NOISES/music/music_list)

music_utt2num_frames=RIRS_NOISES/music/split_utt2num_frames

for f in RIRS_NOISES/simulated_rirs/smallroom/rir_list \
    RIRS_NOISES/simulated_rirs/mediumroom/rir_list \
    RIRS_NOISES/music/music_list \
    RIRS_NOISES/music/split_utt2num_frames \
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

corrupted_data_id=${data_id}_music_corrupted
orig_corrupted_data_id=$corrupted_data_id

if [ $stage -le 1 ]; then
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix="music" \
    --foreground-snrs=$foreground_snrs \
    --background-snrs=$background_snrs \
    --speech-rvb-probability=1 \
    --pointsource-noise-addition-probability=1 \
    --isotropic-noise-addition-probability=1 \
    --num-replications=$num_data_reps \
    --max-noises-per-minute=5 \
    data/${data_id} data/${corrupted_data_id}
fi

corrupted_data_dir=data/${corrupted_data_id}
# Data dir without speed perturbation
orig_corrupted_data_dir=$corrupted_data_dir   

if $speed_perturb; then
  if [ $stage -le 2 ]; then
    for x in $corrupted_data_dir; do
      utils/data/perturb_data_dir_speed_random.sh --speeds "$speeds" $x ${x}_spr
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
  if [ ! -z $feat_suffix ]; then
    utils/copy_data_dir.sh $corrupted_data_dir ${corrupted_data_dir}_$feat_suffix
    corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  fi
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$cmd" --nj $nj --write-utt2num-frames true \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
else
  if [ ! -z $feat_suffix ]; then
    corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  fi
fi 

if [ $stage -le 5 ]; then
  if [ ! -z "$vad_dir" ]; then
    if [ ! -f $vad_dir/speech_labels.scp ]; then
      echo "$0: Could not find file $vad_dir/speech_labels.scp"
      exit 1
    fi
    
    cat $vad_dir/speech_labels.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps "music" | \
      sort -k1,1 > ${corrupted_data_dir}/speech_labels.scp
    
    if [ -f $vad_dir/deriv_weights.scp ]; then
      cat $vad_dir/deriv_weights.scp | \
        steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps "music" | \
        sort -k1,1 > ${corrupted_data_dir}/deriv_weights.scp
    fi
  fi
fi

# music_dir is without speed perturbation 
music_dir=exp/make_music_labels/${orig_corrupted_data_id}
mkdir -p $music_dir

if [ $stage -le 6 ]; then
  if [ ! -f $orig_corrupted_data_dir/additive_signals_info.txt ]; then
    echo "$0: Could not find $orig_corrupted_data_dir/additive_signals_info.txt."
    echo "$0: It is expected to be created by the script reverberate_data_dir.py"
    exit 1
  fi

  splits=
  for n in `seq $nj`; do 
    splits="$splits $music_dir/additive_signals_info.$n.$nj.txt"
  done
  utils/split_scp.pl $orig_corrupted_data_dir/additive_signals_info.txt $splits
  # additive_signals_info.txt is created by the script reverberate_data_dir.py.
  # additive_signals_info.txt is indexed by the recording-id and has the format:
  # <recording-id> list-of-space-separated-tuples
  # where each tuple is written in the format <noise-id>:<start-time>:<duration>
  # It specifies the location where the noise (music) is added and the duration
  # of the noise added. 
  # Note that if the end time of the noise is beyond the duration of the 
  # recording, then it will be truncated.
  utils/data/get_reco2dur.sh $orig_corrupted_data_dir

  awk -v fs=`utils/data/get_frame_shift.sh $corrupted_data_dir` '{print $1" "int($2 / fs)}' \
    $orig_corrupted_data_dir/reco2dur > $orig_corrupted_data_dir/reco2num_frames 

  utils/data/get_utt2num_frames.sh $orig_corrupted_data_dir
  
  $cmd JOB=1:$nj $music_dir/log/get_music_seg.JOB.log \
    segmentation-init-from-additive-signals-info \
      --lengths-rspecifier=ark,t:$orig_corrupted_data_dir/reco2num_frames \
      --additive-signals-segmentation-rspecifier="ark:segmentation-init-from-lengths ark,t:$music_utt2num_frames ark:- |" \
      ark,t:$music_dir/additive_signals_info.JOB.${nj}.txt ark:- \| \
    segmentation-to-ali ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| \
    vector-to-feat ark,t:- ark:- \| \
    extract-feature-segments ark:- $orig_corrupted_data_dir/segments \
      ark:- \| extract-column ark:- ark,t:- \| \
    steps/segmentation/quantize_vector.pl \| \
    segmentation-init-from-ali ark:- ark:- \| \
    segmentation-post-process --merge-adjacent-segments \
      ark:- ark,scp:$music_dir/music_segmentation.JOB.ark,$music_dir/music_segmentation.JOB.scp

  for n in `seq $nj`; do 
    cat $music_dir/music_segmentation.$n.scp
  done > $music_dir/music_segmentation.scp
fi

# Convert label_dir to absolute pathname
mkdir -p $label_dir
label_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $label_dir ${PWD}`
 
if [ $stage -le 7 ]; then
  utils/split_data.sh $corrupted_data_dir $nj
  if $speed_perturb; then
    $cmd JOB=1:$nj $music_dir/log/get_music_labels.JOB.log \
      segmentation-speed-perturb --speeds=0.9:1.0:1.1 ark:$music_dir/music_segmentation.JOB.ark ark:- \| \
      segmentation-to-ali --ignore-missing-lengths --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames ark:- \
      ark,scp:$label_dir/music_labels_${corrupted_data_id}.JOB.ark,$label_dir/music_labels_${corrupted_data_id}.JOB.scp
  else
    $cmd JOB=1:$nj $music_dir/log/get_music_labels.JOB.log \
      segmentation-to-ali --ignore-missing-lengths --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
      ark:$music_dir/music_segmentation.JOB.ark \
      ark,scp:$label_dir/music_labels_${corrupted_data_id}.JOB.ark,$label_dir/music_labels_${corrupted_data_id}.JOB.scp
  fi

  for n in `seq $nj`; do
    cat $label_dir/music_labels_${corrupted_data_id}.$n.scp
  done | \
    utils/filter_scp.pl ${corrupted_data_dir}/utt2spk | sort -k1,1 > ${corrupted_data_dir}/music_labels.scp

  if [ ! -s $corrupted_data_dir/music_labels.scp ]; then
    echo "$0: $corrupted_data_dir/music_labels.scp is empty" && exit 1
  fi

fi

if [ $stage -le 8 ]; then
  utils/split_data.sh --per-utt ${corrupted_data_dir} $nj
  
  cat <<EOF > $music_dir/speech_music_map
0 0 0
0 1 3
1 0 1
1 1 2
EOF

  $cmd JOB=1:$nj $music_dir/log/get_speech_music_labels.JOB.log \
    intersect-int-vectors --mapping-in=$music_dir/speech_music_map --length-tolerance=2 \
    "scp:utils/filter_scp.pl ${corrupted_data_dir}/split${nj}utt/JOB/utt2spk ${corrupted_data_dir}/speech_labels.scp |" \
    "scp:utils/filter_scp.pl ${corrupted_data_dir}/split${nj}utt/JOB/utt2spk ${corrupted_data_dir}/music_labels.scp |" \
    ark,scp:$label_dir/speech_music_labels_${corrupted_data_id}.JOB.ark,$label_dir/speech_music_labels_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do 
    cat $label_dir/speech_music_labels_${corrupted_data_id}.$n.scp
  done > $corrupted_data_dir/speech_music_labels.scp
fi

exit 0
