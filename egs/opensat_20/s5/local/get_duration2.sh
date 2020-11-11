#!/usr/bin/env bash

#/export/c12/aarora8/OpenSAT/16khz_OpenSAT_noise

#1. This script reads noises downloaded from free-sound and converts them into 16khz
#for wav_path_name in /export/c12/aarora8/OpenSAT/OpenSAT_noise/*.wav; do
#  wav_id=$(echo "$wav_path_name" | cut -f7 -d "/")
#  echo $wav_id
#  sox $wav_path_name -r 16000 -c 1 -b 16 /export/c12/aarora8/OpenSAT/16khz_OpenSAT_noise/$wav_id
#done

#2. This part creates wav.scp, utt2spk and spk2utt for the noise extracted and converted to 16khz
#for wav_name in /export/c12/aarora8/OpenSAT/16khz_OpenSAT_noise/*.wav; do
#  recording_id=$(echo "$wav_name" | cut -d"/" -f 7)
#  wav_id=$(echo "$recording_id" | cut -d"." -f 1)
#  echo $wav_id $wav_name >> noise/wav.scp
#  echo $wav_id $wav_id >> noise/utt2spk
#  echo $wav_id $wav_id >> noise/spk2utt
#done

#3. This script creates a text file which have all the time stamps belonging to that wav file which have transcription
#while read -r line;
#  do
#    wav_id=$(echo "$line" | cut -d" " -f 1)
#    grep $wav_id /export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/segments | awk '{print $3 " " $4}' >> data/train/time_stamp/$wav_id
#done < /export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/wav.scp

#4. This script reads a flac file and converts into into wav file
#for wav_path_name in /export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/time_stamp/*; do
#  wav_name=$(echo "$wav_path_name" | cut -f12 -d "/")
#  wav_id=$(echo "$wav_name" | cut -d"." -f 1)
#  wav_id2=$(echo "$wav_id" | cut -c1-33)
#  wav_id3=${wav_id2}mixed
#  echo $wav_id3
#  flac -s -c -d /export/corpora5/opensat_corpora/LDC2019E37/LDC2019E37_SAFE-T_Corpus_Speech_Recording_Audio_Training_Data_R1_V1.1/data/audio/${wav_id3}.flac | sox - -b 16 -t wav -r 16000 -c 1 /export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/wav_files/${wav_id}.wav
#done

#5. This script also reads a flac file and converts into into wav file but do it in a proper way.
#while read -r line;
#  do
#    wav_id=$(echo "$line" | cut -d" " -f 1)
#    wav_path=$(echo "$line" | cut -d" " -f 6)
#    echo $wav_id
#    echo $wav_path
#    flac -s -c -d $wav_path | sox - -b 16 -t wav -r 16000 -c 1 /export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/wav_files2/${wav_id}.wav
#done < /export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/wav.scp

#6. This script is same as 2 but it performs operation on the combined data
#for wav_name in /export/c12/aarora8/OpenSAT/combined_noise/*.wav; do
#  recording_id=$(echo "$wav_name" | cut -d"/" -f 7)
#  wav_id=$(echo "$recording_id" | cut -d"." -f 1)
#  echo $wav_id $wav_name >> /export/c12/aarora8/OpenSAT/combined_noise_wavfile/wav.scp
#  echo $wav_id $wav_id >> /export/c12/aarora8/OpenSAT/combined_noise_wavfile/utt2spk
#  echo $wav_id $wav_id >> /export/c12/aarora8/OpenSAT/combined_noise_wavfile/spk2utt
#done

#copy-vector --binary=false scp:safet_noise_wavfile/vad.scp ark,t:safet_noise_wavfile/vad.txtcopy-vector --binary=false scp:safet_noise_wavfile/vad.scp ark,t:safet_noise_wavfile/vad.txt
#steps/make_mfcc.sh --nj 15 --cmd "$train_cmd" safet_noise_wavfile
#sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd"   safet_noise_wavfile



