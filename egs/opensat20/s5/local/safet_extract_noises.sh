#!/usr/bin/env bash
# Author: Ashish Arora
# Apache 2.0

nj=65
stage=0

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $# != 1 ]; then
   echo "Wrong #arguments ($#, expected 1)"
   echo "Usage: local/safet_extract_noises.sh [options] data/train_safet"
   echo "This script extract noises and use it for augmentation"
   echo "Data, files created = data/train_safet/time_stamp/, data/train_safet/wav_files/ "
   echo "data/local/audio_list  data/safe_t_noise_wavfiles data/safe_t_noise"
   echo "data/safe_t_noise_filtered  " 
  exit 1;
fi

indir=$1

# it creates a text file which have all the time stamps belonging to that wav file
# which have transcription. The $indir/time_stamp/ directory will contain list of
# time segments (where the transcription is present) for wavfile

if [ $stage -le 0 ]; then
  echo "$0: remove old files if they already exist"
  rm -r $indir/time_stamp/ $indir/wav_files/
  rm data/local/audio_list
  rm -r data/safe_t_noise_wavfiles data/safe_t_noise_filtered
  rm -r data/safe_t_noise
fi

if [ $stage -le 0 ]; then
  echo "$0: extracting list of speech segments for each wavfile."
  rm -r $indir/time_stamp/
  mkdir -p $indir/time_stamp/
  while read -r line;
    do
      wav_id=$(echo "$line" | cut -d" " -f 1)
      grep $wav_id $indir/segments | awk '{print $3 " " $4}' >> $indir/time_stamp/$wav_id
  done < $indir/wav.scp
fi

## it reads a flac file and converts into into wav file
if [ $stage -le 1 ]; then
  echo "$0: converting .flac to .wav"
  mkdir -p $indir/wav_files/
  while read -r line;
    do
      wav_id=$(echo "$line" | cut -d" " -f 1)
      wav_path=$(echo "$line" | cut -d" " -f 6)
      echo $wav_id
      echo $wav_path
      flac -s -c -d $wav_path | sox - -b 16 -t wav -r 16000 -c 1 $indir/wav_files/${wav_id}.wav
  done < $indir/wav.scp
fi

## it reads a $indir/wav.scp file and creates audio_list
if [ $stage -le 2 ]; then
  echo "$0: store audiolist (recording id)"
  rm data/local/audio_list
  while read -r line;
    do
      wav_id=$(echo "$line" | cut -d" " -f 1)
      echo $wav_id >> data/local/audio_list
  done < $indir/wav.scp
fi

## it reads a audio wav files, wav time stamps and audio list file and creates noises
if [ $stage -le 3 ]; then
  echo "$0: extract noise from the safet wav files"
  local/safet_extract_noises.py $indir/wav_files $indir/time_stamp data/local/audio_list data/safe_t_noise_wavfiles
fi

# it will give to 10050 noise wav files, its total duration is 55hrs
# utt2spk: <utterance-id> <speaker-id>: noise1 noise1
# wav.scp <recording-id> <wav-path> : noise1 data/safe_t_noise_wavfiles/noise1.wav
# segments:  <utterance-id> <recording-id> <segment-begin> <segment-end> segments: noise1 noise1 0 20
if [ $stage -le 5 ]; then
  echo "$0: create kaldi style data directory for the noise data"
  mkdir -p data/safe_t_noise
  for wav_name in data/safe_t_noise_wavfiles/*.wav; do
    recording_id=$(echo "$wav_name" | cut -d"/" -f 3)
    utt_id=$(echo "$recording_id" | cut -d"." -f 1)
    echo $utt_id $wav_name >> data/safe_t_noise/wav.scp
    echo $utt_id $utt_id >> data/safe_t_noise/utt2spk
    echo $utt_id $utt_id >> data/safe_t_noise/spk2utt
    echo $utt_id $utt_id 0 20 >> data/safe_t_noise/segments
  done
  awk '{ sum += $4 - $3 } END { print sum/3600 }' data/safe_t_noise/segments
  utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" data/safe_t_noise
fi

# # it will apply VAD to noise to find those noise wavfiles which have silence
if [ $stage -le 6 ]; then
  echo "$0: extract features for the safet noise data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 60 data/safe_t_noise
  steps/compute_cmvn_stats.sh data/safe_t_noise
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" data/safe_t_noise
  copy-vector --binary=false scp:data/safe_t_noise/vad.scp ark,t:data/safe_t_noise/vad.txt
fi


if [ $stage -le 7 ]; then
  echo "$0: filter noises using energy based VAD"
  mkdir -p data/safe_t_noise_filtered
  local/get_percent_overlap.py data/safe_t_noise > data/safe_t_noise/percentage_speech
  while read -r line;
  do
    percentage_speech=$(echo "$line" | cut -d" " -f 2)
    uttid=$(echo "$line" | cut -d" " -f 1)
    if [ "$percentage_speech" -gt 80 ]; then
      echo $uttid >> data/safe_t_noise/filtered_noises
    fi
  done < data/safe_t_noise/percentage_speech

  #sort -k2 -n data/safe_t_noise/filtered_noises > data/safe_t_noise/sorted_filtered_noises
  utils/copy_data_dir.sh data/safe_t_noise data/safe_t_noise_filtered
  for f in utt2spk wav.scp feats.scp spk2utt reco2dur cmvn.scp utt2dur utt2num_frames vad.scp; do
    utils/filter_scp.pl data/safe_t_noise/filtered_noises data/safe_t_noise/$f > data/safe_t_noise_filtered/$f
  done
fi
