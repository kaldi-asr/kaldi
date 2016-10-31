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

# copy the wav files and make their  names unique
local/gp_rename_wav_filenames.pl \
    $dir/wav_filenames.txt \
    data/wavs

# get the new wav file names 
local/get_wav_filenames.sh \
    data/wavs \
    $dir

# move the file name list to make them gp specific
mv $dir/wav_filenames.txt $dir/gp_wav_filenames.txt
mv $dir/speaker_directory_paths.txt $dir/gp_speaker_directory_paths.txt

# get a list of the files containing the gp prompts
local/gp_get_prompts_filelist.sh \
    /mnt/corpora/Globalphone/gp/FRF_ASR003/trl

# get the gp prompts 
local/gp_get_prompts.pl \
    data/local/lists/trllist.txt

#condition  the prompts
$lc < $dir/gp_prompts_list.txt | $normalizer | $tokenizer | \
    $deescaper | \
    local/remove.pl | \
    local/gp_reattach_apostrophes.pl | \
	local/gp_oe.pl \
	> \
	$dir/gp_conditioned.txt

paste $dir/gp_id_list.txt $dir/gp_conditioned.txt > $dir/gp_prompts.txt

# put the prompts in their own directory
local/gp_prompts2prompts.pl \
    $dir/gp_wav_filenames.txt \
    $dir/gp_prompts.txt

# get a list of all the speakers
local/get_all_speaker_names.sh data/wavs

# make the list of speakers gp specific
mv $dir/speakers_all.txt $dir/gp_speakers_all.txt

local/gp_get_transcriptions.sh gp

local/get_utt2text.pl \
    $dir/gp_trans_unsorted.txt > \
    $dir/gp_utt2text_unsorted.txt

local/gp_get_utt2spk.sh \
    data/wavs \
    gp

local/gp_get_spk2utt.sh \
    data/wavs \
    gp

local/get_utt2wav_filename.pl \
    data/wavs \
    $dir/gp_speakers_all.txt > \
    $dir/gp_wav_unsorted.scp
exit
mkdir -p data/train

local/sort_transcriptions.sh train
