#!/bin/sh
set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

datadev=$2 # TODO: fix this
audio_path="/export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A/ANALYSIS1/audio/src/"

./cmd.sh                                                                        
./path.sh                                                                       
./utils/parse_options.sh 

mkdir -p $datadev

# 1. create wav.scp, utt2spk, spk2utt files

find $audio_path -name "*.wav" \
  | while read file; do id=$(basename $file | awk '{gsub(".wav","");print}'); \
  echo "$id sox $file -r 8000 -b 16 -c 1 -t wav - |"; done > \
  $datadev/wav.scp

awk '{print $1" "$1}' $datadev/wav.scp > $datadev/utt2spk

cp $datadev/utt2spk $datadev/spk2utt

utils/fix_data_dir.sh $datadev

# 2. segment .wav files
 
# 2.1. create a trivial segments file:

utils/data/get_utt2dur.sh $datadev/

utils/data/get_segments_for_data.sh $datadev/ > $datadev/segments


# 2.2. create uniform segmented directory using: (The durations are in seconds)

utils/data/get_uniform_subsegments.py --max-segment-duration=30 \
--overlap-duration=5 --max-remaining-duration=15 $datadev/segments > \
$datadev/uniform_sub_segments

utils/data/subsegment_data_dir.sh $datadev/ \
  $datadev/uniform_sub_segments $datadev-segmented
