#!/bin/bash
data=$1
dir=$2
# tools
tok_home=/home/tools/mosesdecoder/scripts/tokenizer
lc=$tok_home/lowercase.perl
normalizer="$tok_home/normalize-punctuation.perl -l fr"
tokenizer="$tok_home/tokenizer.perl -l fr"
deescaper=$tok_home/deescape-special-chars.perl

mkdir -p $dir

# first get list of speaker directories
local/get_wav_dirnames.sh $data $dir

# get wav file names
local/get_wav_filenames.sh $data $dir

# move the files to make them sri_gabon specific
mv $dir/wav_filenames.txt $dir/sri_gabon_wav_filenames.txt
mv $dir/speaker_directory_paths.txt $dir/sri_gabon_speaker_directory_paths.txt

# get only the wav files for read speech
local/sri_gabon_get_read_wav_filenames.pl \
    $dir/sri_gabon_wav_filenames.txt > \
    $dir/sri_gabon_read_wav_filenames.txt

# get only the wav files for conversational speech
local/sri_gabon_get_conv_wav_filenames.pl \
    $dir/sri_gabon_wav_filenames.txt > \
    $dir/sri_gabon_conv_wav_filenames.txt

# copy the wav files and make their  names unique
local/sri_gabon_rename_wav_filenames.pl $dir/sri_gabon_wav_filenames.txt data/wavs

# get the new wav file names 
local/sri_gabon_get_wav_filenames.sh data/wavs $dir

# move the files to make them sri_gabon specific
mv $dir/wav_filenames.txt $dir/sri_gabon_wav_filenames.txt

#  get list of speaker directories again
local/sri_gabon_get_wav_dirnames.sh data/wavs $dir

# move the file to make it sri_gabon specific 
mv $dir/speaker_directory_paths.txt $dir/sri_gabon_speaker_directory_paths.txt

# get all speaker names
local/sri_gabon_get_all_speaker_names.sh data/wavs

mv $dir/speakers_all.txt $dir/sri_gabon_speakers_all.txt

local/get_speaker_names.pl \
    $dir/sri_gabon_speaker_directory_paths.txt \
    sri_gabon > \
    $dir/sri_gabon_speaker_names.txt

# get   randomly  selected prompts for sri_gabon
local/sri_gabon_make_randomly_selected_transcription_file.sh

#cut -f 1 data/local/tmp/sri_gabon_prompts.txt> $dir/sri_gabon_id.txt
# use the file names as indices to the prompts
local/sri_gabon_get_file_names.pl \
    $dir/sri_gabon_wav_filenames.txt > \
    $dir/sri_gabon_id.txt
cut -f 2 data/local/tmp/sri_gabon_prompts.txt > $dir/sri_gabon_sents.txt

# condition the sri_gabon prompts
$lc < $dir/sri_gabon_sents.txt | $normalizer | $tokenizer | \
    $deescaper | local/remove.pl > $dir/sri_gabon_conditioned.txt

# put the prompts back together with indices
paste $dir/sri_gabon_id.txt $dir/sri_gabon_conditioned.txt > \
      $dir/sri_gabon_prompts.txt

# put the labels in their own directory
local/sri_gabon_prompts2prompts.pl \
    $dir/sri_gabon_wav_filenames.txt \
    $dir/sri_gabon_prompts.txt

# put the transcriptions in 1 file
local/get_transcriptions.sh sri_gabon

local/sri_gabon_get_utt2text.pl \
    $dir/sri_gabon_trans_unsorted.txt > \
    $dir/sri_gabon_utt2text_unsorted.txt

local/get_utt2spk.sh data/wavs sri_gabon

local/get_spk2utt.sh data/wavs sri_gabon

local/get_utt2wav_filename.pl \
    data/wavs \
    $dir/sri_gabon_speaker_names.txt > \
    $dir/sri_gabon_wav_unsorted.scp

mkdir -p data/sri_gabon

fld=sri_gabon
localdata=data/local
localtmp=$localdata/tmp/sri_gabon
outdir=data
sort  -u $dir/${fld}_wav_unsorted.scp > ${outdir}/${fld}/wav.scp
sort -u  $dir/${fld}_spk2utt_unsorted.txt > ${outdir}/${fld}/spk2utt
sort -u  $dir/${fld}_utt2spk_unsorted.txt > ${outdir}/${fld}/utt2spk
sort -u  $dir/${fld}_utt2text_unsorted.txt > ${outdir}/${fld}/filename2text
sort -u  $dir/${fld}_utt2text_unsorted.txt > ${outdir}/${fld}/text
