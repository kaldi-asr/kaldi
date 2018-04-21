#!/bin/sh
set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

datadev=$1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
. ./lang.conf

mkdir -p $datadev

# 1. create the reference transcript $datadev/reftext

if [ $(basename $datadev) == 'analysis1' ]; then
  ls -d ${audio_path_analysis1}/transcription/* > "list.tmp"
  rm -rf {zero,brac,all}.tmp
  rm -rf all.tmp.sort

  while read line; do
    h=$(head -1 "$line")
    if [[ $h == "0"* ]]
    then                                                                     
      # starts with 0.000"
      echo $line | while read ref; do
        s=${ref##*/}
        awk '{l[NR] = $0} END {for (i=1; i<=NR-1; i++) print l[i]}' "$ref" | \
          cut -f1,2 --complement | tr '\n' ' ' | \
          awk '{$0="'${s%.transcription.txt}' " $0}1' >> zero.tmp;
      done
    else
      # starts with [0.000]"
      echo $line | while read ref; do
        s=${ref##*/}
        awk 'NR%2==0' "$ref" | tr '\n' ' ' | \
          awk '{$0="'${s%.transcription.txt}' " $0}1'>> brac.tmp;
      done
    fi
  done < "list.tmp"

  cat zero.tmp brac.tmp > all.tmp
  mv all.tmp $datadev/reftext
  rm -rf {zero,brac,list}.tmp
fi

# 2. create wav.scp, utt2spk, spk2utt files

wav_path=
if [ $(basename $datadev) == 'analysis1' ]; then
  wav_path=${audio_path_analysis1}/src
elif [ $(basename $datadev) == 'test_dev' ]; then
  wav_path=${audio_path_dev}/src
elif [ $(basename $datadev) == 'eval1' ]; then
  wav_path=${audio_path_eval1}/src
elif [ $(basename $datadev) == 'eval2' ]; then
  wav_path=${audio_path_eval2}/src
fi
[ -z ${wav_path} ] && echo "$0: test data should be either analysis1, test_dev, eval1 or eval2." && exit 1

find ${wav_path} -name "*.wav" \
  | while read file; do id=$(basename $file | awk '{gsub(".wav","");print}'); \
  echo "$id sox $file -r 8000 -b 16 -c 1 -t wav - |"; done > \
  $datadev/wav.scp

awk '{print $1" "$1}' $datadev/wav.scp > $datadev/utt2spk

cp $datadev/utt2spk $datadev/spk2utt

utils/fix_data_dir.sh $datadev

# 3. segment .wav files
 
# 3.1. create a trivial segments file:

utils/data/get_utt2dur.sh $datadev/

utils/data/get_segments_for_data.sh $datadev/ > $datadev/segments


# 3.2. create uniform segmented directory using: (The durations are in seconds)

utils/data/get_uniform_subsegments.py --max-segment-duration=30 \
--overlap-duration=5 --max-remaining-duration=15 $datadev/segments > \
$datadev/uniform_sub_segments

utils/data/subsegment_data_dir.sh $datadev/ \
  $datadev/uniform_sub_segments ${datadev}_segmented

