#!/usr/bin/env bash

#/export/c12/aarora8/OpenSAT/16khz_OpenSAT_noise

#for wav_path_name in /export/c05/aarora8/kaldi2/egs/icsisafet/s5/audio/*.flac; do
#  wav_id_flac=$(echo "$wav_path_name" | cut -f10 -d "/")
#  wav_id=$(echo "$wav_id_flac" | cut -d"." -f 1)
#  echo $wav_id
#  sox $wav_path_name -t wav -r 16000 -c 1 -b 16 audio_wav/${wav_id}.wav
#done

#for wav_name in ls /export/c12/aarora8/OpenSAT/16khz_OpenSAT_noise/*.wav; do
#  recording_id=$(echo "$wav_name" | cut -d"/" -f 7)
#  wav_id=$(echo "$recording_id" | cut -d"." -f 1)
#  echo $wav_id $wav_name >> noise/wav.scp
#  echo $wav_id $wav_id >> noise/utt2spk
#  echo $wav_id $wav_id >> noise/spk2utt
#done

# 000001_0000000_0000112_0001_20190122_200119_part1_train_xx /export/c02/aarora8/kaldi2/egs/icsisafet/s5/data/ihm/train_isa/data/raw_mfcc_train_isa.1.ark:59
#cut -d"/" -f 11 data/ihm/train_aug/feats.scp > data/ihm/train_aug/data_loca1
#cat data/ihm/train_aug/data_loca1 | sort | uniq | head
#while read -r line;
#  do
#    data_dir=$(echo "$line" | cut -d"/" -f 11)
#    echo $data_dir >> data/ihm/train_aug/data_loca
#done < data/ihm/train_aug/feats.scp

# get wav file path and names in r11
#while read -r line;
#  do
#    data_dir=$(echo "$line" | cut -d" " -f 6)
#    echo $data_dir  >> /export/c05/aarora8/kaldi2/egs/icsisafet/s5/wav_safe_t_r11
#done < /export/c05/aarora8/kaldi2/egs/icsisafet/s5/meta_dexp/2G_sp_amiicsisafet/data/safe_t_r11/wav.scp

# get duration of each wav file
#while read -r line;
#  do
#    soxi -D $line
#done < /export/c05/aarora8/kaldi2/egs/icsisafet/s5/wav_safe_t_r11

#
#while read -r line;
#  do
#    data_dir=$(echo "$line" | cut -d" " -f 6)
#    echo $data_dir >> /export/c05/aarora8/kaldi2/egs/icsisafet/s5/wav_safe_t_r20
#done < /export/c05/aarora8/kaldi2/egs/icsisafet/s5/meta_dexp/2G_sp_amiicsisafet/data/safe_t_r20/wav.scp
#
#while read -r line;
#  do
#    soxi -D $line
#done < /export/c05/aarora8/kaldi2/egs/icsisafet/s5/wav_safe_t_r20

#for wav_name in /export/corpora5/opensat_corpora/LDC2019E37/LDC2019E37_SAFE-T_Corpus_Speech_Recording_Audio_Training_Data_R1_V1.1/data/audio/*.flac; do
#soxi -D $wav_name
#done

for wav_name in /export/corpora5/opensat_corpora/LDC2020E10/LDC2020E10_SAFE-T_Corpus_Speech_Recording_Audio_Training_Data_R2/data/audio/*.flac; do
soxi -D $wav_name
done

