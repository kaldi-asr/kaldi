#!/bin/sh
set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
. ./lang.conf

datadev=$1

mkdir -p $datadev

# 1. create the reference transcript $datadev/reftext

dataset=$(basename $datadev)

audio_path=
if [ $dataset == "analysis1" ]; then
  audio_path=${audio_path_analysis1}
elif [ $dataset == "analysis2" ]; then
  audio_path=${audio_path_analysis2}
elif [ $(basename $datadev) == 'test_dev' ]; then
  audio_path=${audio_path_dev}
elif [ $(basename $datadev) == 'eval1' ]; then
  audio_path=${audio_path_eval1}
elif [ $(basename $datadev) == 'eval2' ]; then
  audio_path=${audio_path_eval2}
elif [ $(basename $datadev) == 'eval3' ]; then
  audio_path=${audio_path_eval3}
fi

[ -z ${audio_path} ] && echo "$0: test data should be either analysis1, analysis2, test_dev, eval1 or eval2." && exit 1

metadata_file=${audio_path}/metadata/metadata.tsv

if [ $stage -le 0 ]; then
  mkdir -p data/local/$dataset

  tail -n +2 $metadata_file | \
    perl -ane '$F[0] =~ s/.wav//; print "$F[0] $F[1]\n";' > \
    data/local/$dataset/all_list

  awk '{if ($2 == "CS") { print $1 } }' data/local/$dataset/all_list > data/local/$dataset/call_list
  awk '{if ($2 != "CS") { print $1 } }' data/local/$dataset/all_list > data/local/$dataset/non_call_list
fi

if [ $stage -le 2 ]; then
  rm data/local/$dataset/{wav.scp,reco2file_and_channel} 2>/dev/null || true

  if [ $dataset == "analysis1" ] || [ $dataset == "analysis2" ]; then
    local/parse_dev_transcripts.py $audio_path \
      data/local/$dataset/call_list \
      data/local/$dataset/non_call_list \
      data/local/$dataset
  else
    for f in $(cat data/local/$dataset/call_list); do
      wav_file="$audio_path/src/$f.wav"

      echo "${f}_inLine sox $wav_file -r 8000 -b 16 -c 1 -t wav - remix 1 |" >> data/local/$dataset/wav.scp
      echo "${f}_outLine sox $wav_file -r 8000 -b 16 -c 1 -t wav - remix 2 |" >> data/local/$dataset/wav.scp
      echo "${f}_inLine ${f} A" >> data/local/$dataset/reco2file_and_channel
      echo "${f}_outLine ${f} B" >> data/local/$dataset/reco2file_and_channel
    done
    
    for f in $(cat data/local/$dataset/non_call_list); do
      wav_file="$audio_path/src/$f.wav"

      echo "${f} sox $wav_file -r 8000 -b 16 -c 1 -t wav - |" >> data/local/$dataset/wav.scp
      echo "${f} ${f} 1" >> data/local/$dataset/reco2file_and_channel
    done

    awk '{print $1" "$1}' data/local/$dataset/wav.scp > data/local/$dataset/utt2spk
  fi
  utils/utt2spk_to_spk2utt.pl data/local/$dataset/utt2spk > data/local/$dataset/spk2utt
  utils/fix_data_dir.sh data/local/$dataset
  
  utils/copy_data_dir.sh data/local/$dataset $datadev
fi

if [ $stage -le 3 ]; then
  if [ $dataset == "analysis1" ] || [ $dataset == "analysis2" ]; then
    cat data/local/$dataset/all_list | awk '{print $1" <"$2",O>"}' > \
      data/local/$dataset/all_list_labels
    
    awk '{print $2" "$1" "$3" "$4" "$1}' $datadev/segments | \
      utils/apply_map.pl -f 1 $datadev/reco2file_and_channel | \
      utils/apply_map.pl -f 3 $datadev/utt2spk | \
      awk '{print $1" "$2" "$3" "$4" "$5" "$1" "$6}' | \
      utils/apply_map.pl -f 7 $datadev/text | \
      utils/apply_map.pl -f 6 data/local/$dataset/all_list_labels | \
      sort +0 -1 +1 -2 +3nb -4 > \
      $datadev/stm

    touch $datadev/glm
  fi
fi

# 3. segment .wav files
 
# 3.1. create a trivial segments file:

if [ $stage -le 4 ]; then
  utils/data/get_utt2dur.sh --nj 4 --cmd "$train_cmd" ${datadev}

  if [ ! -f $datadev/segments ]; then
    utils/data/get_segments_for_data.sh $datadev/ > $datadev/segments
  fi

  # 3.2. create uniform segmented directory using: (The durations are in seconds)

  if [ $dataset == "analysis1" ] || [ $dataset == "analysis2" ]; then
    utils/data/convert_data_dir_to_whole.sh $datadev ${datadev}_whole
    utils/data/get_utt2dur.sh --nj 4 --cmd "$train_cmd" ${datadev}_whole
    
    utils/data/get_segments_for_data.sh ${datadev}_whole > ${datadev}_whole/segments
    utils/data/get_uniform_subsegments.py --max-segment-duration=30 \
    --overlap-duration=5 --max-remaining-duration=15 ${datadev}_whole/segments > \
    ${datadev}_whole/uniform_sub_segments

    utils/data/subsegment_data_dir.sh ${datadev}_whole/ \
      ${datadev}_whole/uniform_sub_segments ${datadev}_segmented
  else
    utils/data/get_uniform_subsegments.py --max-segment-duration=30 \
    --overlap-duration=5 --max-remaining-duration=15 ${datadev}/segments > \
    ${datadev}/uniform_sub_segments

    utils/data/subsegment_data_dir.sh ${datadev}/ \
      ${datadev}/uniform_sub_segments ${datadev}_segmented
  fi
fi
