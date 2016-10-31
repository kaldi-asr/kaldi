#!/bin/bash
data=$1
dir=$2
# tools
tok_home=/home/tools/mosesdecoder/scripts/tokenizer
lowercaser=$tok_home/lowercase.perl
normalizer="$tok_home/normalize-punctuation.perl -l fr"
tokenizer="$tok_home/tokenizer.perl -l fr"
deescaper=$tok_home/deescape-special-chars.perl

mkdir -p $dir

# first get list of speaker directories
local/get_wav_dirnames.sh $data $dir

# get wav file names
local/get_wav_filenames.sh $data $dir

# move the file name list to make them test specific
mv $dir/wav_filenames.txt $dir/test_wav_filenames.txt
mv $dir/speaker_directory_paths.txt $dir/test_speaker_directory_paths.txt

cut -f 1 local/src/central_accord_prompts.txt > $dir/central_accord_prompts_id.txt
cut -f 2 local/src/central_accord_prompts.txt > $dir/central_accord_prompts_sent.txt
$lowercaser < $dir/central_accord_prompts_sent.txt | $normalizer  | $tokenizer |$deescaper | local/remove.pl > $dir/test_prompts_sent.txt
paste $dir/central_accord_prompts_id.txt $dir/test_prompts_sent.txt > $dir/test_prompts.txt
local/central_accord_prompts2prompts4speaker.pl $dir/test_wav_filenames.txt $dir/test_prompts.txt
local/get_all_speaker_names.sh $data

cp $dir/speakers_all.txt $dir/test_speakers_all.txt

local/get_speaker_names.pl $dir/test_speaker_directory_paths.txt test > $dir/test_speaker_names.txt

local/get_transcriptions.sh test

local/get_utt2text.pl $dir/test_trans_unsorted.txt > $dir/test_utt2text_unsorted.txt

local/get_utt2spk.sh $data test

local/get_spk2utt.sh $data test

local/get_utt2wav_filename.pl $data $dir/test_speaker_names.txt > $dir/test_wav_unsorted.scp

mkdir -p data/test

local/sort_transcriptions.sh test
