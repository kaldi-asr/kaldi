#!/bin/bash
set -e
set -u
set -o pipefail

. path.sh
. cmd.sh

num_data_reps=5
data_dir=data/train_si284

nj=40
reco_nj=40

stage=0
corruption_stage=-10

pad_silence=false

mfcc_config=conf/mfcc_hires_bp_vh.conf
feat_suffix=hires_bp_vh
mfcc_irm_config=conf/mfcc_hires_bp.conf

dry_run=false
corrupt_only=false
speed_perturb=true

reco_vad_dir=

max_jobs_run=20

foreground_snrs="5:2:1:0:-2:-5:-10:-20"
background_snrs="5:2:1:0:-2:-5:-10:-20"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

data_id=`basename ${data_dir}`

rvb_opts=()
# This is the config for the system using simulated RIRs and point-source noises
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
rvb_opts+=(--noise-set-parameters RIRS_NOISES/music/music_list)

music_utt2num_frames=RIRS_NOISES/music/split_utt2num_frames

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

if $dry_run; then
  exit 0
fi

corrupted_data_dir=data/${corrupted_data_id}
# Data dir without speed perturbation
orig_corrupted_data_dir=$corrupted_data_dir   

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
  if [ ! -z $feat_suffix ]; then
    utils/copy_data_dir.sh $corrupted_data_dir ${corrupted_data_dir}_$feat_suffix
    corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  fi
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
else
  if [ ! -z $feat_suffix ]; then
    corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  fi
fi 

if [ $stage -le 8 ]; then
  if [ ! -z "$reco_vad_dir" ]; then
    if [ ! -f $reco_vad_dir/speech_labels.scp ]; then
      echo "$0: Could not find file $reco_vad_dir/speech_labels.scp"
      exit 1
    fi
    
    cat $reco_vad_dir/speech_labels.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps "music" | \
      sort -k1,1 > ${corrupted_data_dir}/speech_labels.scp
    
    cat $reco_vad_dir/deriv_weights.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps "music" | \
      sort -k1,1 > ${corrupted_data_dir}/deriv_weights.scp
  fi
fi

# music_dir is without speed perturbation 
music_dir=exp/make_music_labels/${orig_corrupted_data_id}
music_data_dir=$music_dir/music_data

mkdir -p $music_data_dir

if [ $stage -le 10 ]; then
  utils/data/get_reco2num_frames.sh --nj $reco_nj $orig_corrupted_data_dir
  utils/split_data.sh --per-reco ${orig_corrupted_data_dir} $reco_nj

  cp $orig_corrupted_data_dir/wav.scp $music_data_dir
  
  # The first rspecifier is a dummy required to get the recording-id as key.
  # It has no segments in it as they are all removed by --remove-labels.
  $train_cmd JOB=1:$reco_nj $music_dir/log/get_music_seg.JOB.log \
    segmentation-init-from-additive-signals-info --lengths-rspecifier=ark,t:${orig_corrupted_data_dir}/reco2num_frames \
    --additive-signals-segmentation-rspecifier="ark:segmentation-init-from-lengths ark:$music_utt2num_frames ark:- |" \
    "ark,t:utils/filter_scp.pl ${orig_corrupted_data_dir}/split${reco_nj}reco/JOB/reco2utt $orig_corrupted_data_dir/additive_signals_info.txt |" \
    ark:- \| \
    segmentation-post-process --merge-adjacent-segments ark:- \
    ark:- \| \
    segmentation-to-segments ark:- ark:$music_data_dir/utt2spk.JOB \
    $music_data_dir/segments.JOB

  utils/data/get_reco2utt.sh $corrupted_data_dir
  for n in `seq $reco_nj`; do cat $music_data_dir/utt2spk.$n; done > $music_data_dir/utt2spk
  for n in `seq $reco_nj`; do cat $music_data_dir/segments.$n; done > $music_data_dir/segments
  
  utils/fix_data_dir.sh $music_data_dir

  if $speed_perturb; then
    utils/data/perturb_data_dir_speed_3way.sh $music_data_dir ${music_data_dir}_spr
    mv ${music_data_dir}_spr/segments{,.temp}
    cat ${music_data_dir}_spr/segments.temp | \
      utils/filter_scp.pl -f 2 ${corrupted_data_dir}/reco2utt > ${music_data_dir}_spr/segments
    utils/fix_data_dir.sh ${music_data_dir}_spr
    rm ${music_data_dir}_spr/segments.temp
  fi
fi

if $speed_perturb; then
  music_data_dir=${music_data_dir}_spr
fi

label_dir=music_labels

mkdir -p $label_dir
label_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $label_dir ${PWD}`

if [ $stage -le 11 ]; then
  utils/split_data.sh --per-reco ${corrupted_data_dir} $reco_nj
  # TODO: Don't assume that its whole data directory.
  nj=$reco_nj
  if [ $nj -gt 4 ]; then
    nj=4
  fi
  utils/data/get_utt2num_frames.sh --cmd "$train_cmd" --nj $nj ${corrupted_data_dir} 
  utils/data/get_reco2utt.sh $music_data_dir/

  $train_cmd JOB=1:$reco_nj $music_dir/log/get_music_labels.JOB.log \
    segmentation-init-from-segments --shift-to-zero=false \
    "utils/filter_scp.pl -f 2 ${corrupted_data_dir}/split${reco_nj}reco/JOB/reco2utt ${music_data_dir}/segments |" ark:- \| \
    segmentation-combine-segments-to-recordings ark:- \
    "ark,t:utils/filter_scp.pl ${corrupted_data_dir}/split${reco_nj}reco/JOB/reco2utt ${music_data_dir}/reco2utt |" \
    ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- \
    ark,scp:$label_dir/music_labels_${corrupted_data_id}.JOB.ark,$label_dir/music_labels_${corrupted_data_id}.JOB.scp
fi

for n in `seq $reco_nj`; do
  cat $label_dir/music_labels_${corrupted_data_id}.$n.scp
done | utils/filter_scp.pl ${corrupted_data_dir}/utt2spk > ${corrupted_data_dir}/music_labels.scp

if [ $stage -le 12 ]; then
  utils/split_data.sh --per-reco ${corrupted_data_dir} $reco_nj
  
  cat <<EOF > $music_dir/speech_music_map
0 0 0
0 1 3
1 0 1
1 1 2
EOF

  $train_cmd JOB=1:$reco_nj $music_dir/log/get_speech_music_labels.JOB.log \
    intersect-int-vectors --mapping-in=$music_dir/speech_music_map \
    "scp:utils/filter_scp.pl ${corrupted_data_dir}/split${reco_nj}reco/JOB/reco2utt ${corrupted_data_dir}/speech_labels.scp |" \
    "scp:utils/filter_scp.pl ${corrupted_data_dir}/split${reco_nj}reco/JOB/reco2utt ${corrupted_data_dir}/music_labels.scp |" \
    ark,scp:$label_dir/speech_music_labels_${corrupted_data_id}.JOB.ark,$label_dir/speech_music_labels_${corrupted_data_id}.JOB.scp

  for n in `seq $reco_nj`; do 
    cat $label_dir/speech_music_labels_${corrupted_data_id}.$n.scp
  done > $corrupted_data_dir/speech_music_labels.scp
fi

exit 0
