#!/usr/bin/env bash

## Only run this file from the example root directory
##      $ ./local/data_prep.sh

mkdir -p "data/train" "data/test" "data/local"

source ./path.sh

# Dimex100 unziped corpus root directory
CORPUS_DIR="$1"

# Corpus data
#
#   Number of Different speakers:   100
#   Speakers common utterances:     10
#   Speakers individual utterances: 50
#
# Training/testing split
#
#   Common utterances for training:     10 (100%)
#   Individual utterances for training: 40 (80%)
#   Individual utterances for testing:  10 (20%)
N_SPEAKERS=100
N_COMMON_UTTERANCES=10
N_INDIVIDUAL_UTTERANCES=50
N_INDIVIDUAL_UTTERANCES_TRAINING=40
N_INDIVIDUAL_UTTERANCES_TESTING=10

# speakerId-utteranceId-[c|i]
#   c = speaker common utterances (10)
#   i = speaker individual utterances (50)
#
#   e.g.:
#       s001-01-c
#       ...
#       s001-10-c
#       ...
#       s001-01-i
#       ...
#       s001-50-i

## 80-20 train-test split
## Only individual utterances are used in testing
#    10/10 common utterances go into training
#    40/50 individual utterances go into training
#    10/50 individual utterances go into testing

function make_speaker_id
{
    printf "s%03d" "$1"
}

function make_sentence_id
{
    printf "%02d" "$1"
}

#####################################
# Convert wave audio to 16-bit, 16kHz
#####################################

function convert_to_16khz
{
    for i in $(seq 1 $N_SPEAKERS); do
        speaker_id=$(make_speaker_id $i)

        mkdir -p "$CORPUS_DIR/$speaker_id/audio_16k/comunes"
        mkdir -p "$CORPUS_DIR/$speaker_id/audio_16k/individuales"

        # Common utterances
        for j in $(seq 1 $N_COMMON_UTTERANCES); do
            sentence_id=$(make_sentence_id $j)
            old_wav_file="$CORPUS_DIR/$speaker_id/audio_editado/comunes/$speaker_id$sentence_id.wav"
            new_wav_file="$CORPUS_DIR/$speaker_id/audio_16k/comunes/$speaker_id$sentence_id.wav"
            sox "$old_wav_file" -r 16k "$new_wav_file"
        done

        # Individual utterances
        for k in $(seq 1 $N_INDIVIDUAL_UTTERANCES_TRAINING); do
            sentence_id=$(make_sentence_id $k)
            old_wav_file="$CORPUS_DIR/$speaker_id/audio_editado/individuales/$speaker_id$sentence_id.wav"
            new_wav_file="$CORPUS_DIR/$speaker_id/audio_16k/individuales/$speaker_id$sentence_id.wav"
            sox "$old_wav_file" -r 16k "$new_wav_file"
        done
    done
}

if [[ ! -d "$CORPUS_DIR/s001/audio_16k" ]]; then
    echo
    echo Converting audio from 44.1kHz to 16kHz
    echo
    convert_to_16khz
fi



#################
# data/train/text
# data/test/text
#################

# speakerId-utteranceId-[c|i]
#   c = speaker common utterances (10)
#   i = speaker individual utterances (50)
#
#   e.g.:
#       s001-01-c
#       ...
#       s001-10-c
#       ...
#       s001-01-i
#       ...
#       s001-50-i

## 80-20 train-test split
## Only individual utterances are used in testing
#    10/10 common utterances go into training
#    40/50 individual utterances go into training
#    10/50 individual utterances go into testing



function clean
{
    echo "$1" \
        | tr -d '\r' \
        | tr '[:upper:]' '[:lower:]' \
        | sed \
            -e 's/á/a/g' -e 's/é/e/g' -e 's/í/i/g' -e 's/ó/o/g' -e 's/ú/u/g' \
            -e 's/Á/a/g' -e 's/É/e/g' -e 's/Í/i/g' -e 's/Ó/o/g' -e 's/Ú/u/g' \
            -e 's/ñ/n/g' -e 's/Ñ/n/g' -e 's/ü/u/g' -e 's/Ü/u/g' \
        | tr -d -c "a-zA-Z0-9 \r\n"
        # | tr -d -c "_,.;:\-?¿!'\"()" \
}

### Generate data/train/text
for i in $(seq 1 $N_SPEAKERS); do
    speaker_id=$(make_speaker_id $i)

    # Common utterances
    for j in $(seq 1 $N_COMMON_UTTERANCES); do
        sentence_id=$(make_sentence_id $j)
        utterance_id="$speaker_id-$sentence_id-c"
        trans_file="$CORPUS_DIR/$speaker_id/texto/comunes/$speaker_id$sentence_id.txt"
        iconv -f iso-8859-1 -t utf-8 "$trans_file" > "$trans_file.utf8"
        if [ -f "$trans_file.utf8" ]; then
            transcription=$(cat "$trans_file.utf8")
            transcription=$(clean "$transcription")
            echo "$utterance_id $transcription" >> "data/train/text"
        fi
    done

    # Individual utterances
    for k in $(seq 1 $N_INDIVIDUAL_UTTERANCES_TRAINING); do
        sentence_id=$(make_sentence_id $k)
        utterance_id="$speaker_id-$sentence_id-i"
        trans_file="$CORPUS_DIR/$speaker_id/texto/individuales/$speaker_id$sentence_id.txt"
        iconv -f iso-8859-1 -t utf-8 "$trans_file" > "$trans_file.utf8"
        if [ -f "$trans_file.utf8" ]; then
            transcription=$(cat "$trans_file.utf8")
            transcription=$(clean "$transcription")
            echo "$utterance_id $transcription" >> "data/train/text"
        fi
    done

done


### Generate data/test/text
for i in $(seq 1 $N_SPEAKERS); do
    speaker_id=$(make_speaker_id $i)

    # Individual utterances
    for k in $(seq $N_INDIVIDUAL_UTTERANCES_TRAINING $N_INDIVIDUAL_UTTERANCES); do
        sentence_id=$(make_sentence_id $k)
        utterance_id="$speaker_id-$sentence_id-i"
        trans_file="$CORPUS_DIR/$speaker_id/texto/individuales/$speaker_id$sentence_id.txt"
        iconv -f iso-8859-1 -t utf-8 "$trans_file" > "$trans_file.utf8"
        if [ -f "$trans_file.utf8" ]; then
            transcription=$(cat "$trans_file.utf8")
            transcription=$(clean "$transcription")
            echo "$utterance_id $transcription" >> "data/test/text"
        fi
    done

done




####################
# data/train/wav.scp
# data/test/wav.scp
####################


### Generate data/train/wav.scp
for i in $(seq 1 $N_SPEAKERS); do
    speaker_id=$(make_speaker_id $i)

    # Common utterances
    for j in $(seq 1 $N_COMMON_UTTERANCES); do
        sentence_id=$(make_sentence_id $j)
        utterance_id="$speaker_id-$sentence_id-c"
        wav_file="$CORPUS_DIR/$speaker_id/audio_16k/comunes/$speaker_id$sentence_id.wav"
        if [ -f "$wav_file" ]; then
            echo "$utterance_id $wav_file" >> "data/train/wav.scp"
        fi
    done

    # Individual utterances
    for k in $(seq 1 $N_INDIVIDUAL_UTTERANCES_TRAINING); do
        sentence_id=$(make_sentence_id $k)
        utterance_id="$speaker_id-$sentence_id-i"
        wav_file="$CORPUS_DIR/$speaker_id/audio_16k/individuales/$speaker_id$sentence_id.wav"
        if [ -f "$wav_file" ]; then
            echo "$utterance_id $wav_file" >> "data/train/wav.scp"
        fi
    done

done


### Generate data/test/wav.scp
for i in $(seq 1 $N_SPEAKERS); do
    speaker_id=$(make_speaker_id $i)

    # Individual utterances
    for k in $(seq $N_INDIVIDUAL_UTTERANCES_TRAINING $N_INDIVIDUAL_UTTERANCES); do
        sentence_id=$(make_sentence_id $k)
        utterance_id="$speaker_id-$sentence_id-i"
        wav_file="$CORPUS_DIR/$speaker_id/audio_16k/individuales/$speaker_id$sentence_id.wav"
        if [ -f "$wav_file" ]; then
            echo "$utterance_id $wav_file" >> "data/test/wav.scp"
        fi
    done

done




####################
# data/train/utt2spk
# data/test/utt2spk
####################

# Take IDs from 'text' file to avoid including missing data's IDs

### Generate data/train/utt2spk
utterance_ids=$(cat "data/train/text" | cut -d' ' -f1)

while read -r utterance_id; do
    speaker_id=$(echo "$utterance_id" | cut -d'-' -f1)
    echo "$utterance_id $speaker_id" >> "data/train/utt2spk"
done <<< "$utterance_ids"


### Generate data/test/utt2spk
utterance_ids=$(cat "data/test/text" | cut -d' ' -f1)

while read -r utterance_id; do
    speaker_id=$(echo "$utterance_id" | cut -d'-' -f1)
    echo "$utterance_id $speaker_id" >> "data/test/utt2spk"
done <<< "$utterance_ids"


############
# Sort files
############

LC_ALL=C sort -o "data/train/text" "data/train/text"
LC_ALL=C sort -o "data/test/text" "data/test/text"
LC_ALL=C sort -o "data/train/wav.scp" "data/train/wav.scp"
LC_ALL=C sort -o "data/test/wav.scp" "data/test/wav.scp"
LC_ALL=C sort -o "data/train/utt2spk" "data/train/utt2spk"
LC_ALL=C sort -o "data/test/utt2spk" "data/test/utt2spk"


####################
# data/train/spk2utt
# data/test/spk2utt
####################
utils/utt2spk_to_spk2utt.pl "data/train/utt2spk" > "data/train/spk2utt"
utils/utt2spk_to_spk2utt.pl "data/test/utt2spk" > "data/test/spk2utt"
