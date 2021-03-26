#! /bin/bash

# Copyright Johns Hopkins University
#   2019 Fei Wu


printf "\t File Check in folder: %s.\n" "$1"

WavScp="$1/wav.scp"
Text="$1/text"
Utt2Spk="$1/utt2spk"
Gend="$1/utt2gender"
Spk2Utt="$1/spk2utt"
rm -f $WavScp $Text $Utt2Spk $Gend $Spk2Utt



