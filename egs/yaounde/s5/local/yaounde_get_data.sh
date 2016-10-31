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

# move the files to make them yaounde specific
mv $dir/wav_filenames.txt $dir/yaounde_wav_filenames.txt
mv $dir/speaker_directory_paths.txt $dir/yaounde_speaker_directory_paths.txt

# get the yaounde prompts
cut -f 1 local/src/yaounde_read_prompts.txt > $dir/yaounde_prompts_id.txt
cut -f 2 local/src/yaounde_read_prompts.txt > $dir/yaounde_sents.txt

$lowercaser < $dir/yaounde_sents.txt | $normalizer | $tokenizer | $deescaper | local/remove.pl > $dir/yaounde_prompts_lc_norm_tok_esc_nopunc.txt
paste $dir/yaounde_prompts_id.txt $dir/yaounde_prompts_lc_norm_tok_esc_nopunc.txt > $dir/yaounde_prompts.txt

local/prompts2prompts4speaker.pl $dir/yaounde_wav_filenames.txt $dir/yaounde_prompts.txt

local/get_all_speaker_names.sh $data

mv $dir/speakers_all.txt $dir/yaounde_speakers_all.txt

local/get_speaker_names.pl $dir/yaounde_speaker_directory_paths.txt yaounde > $dir/yaounde_speaker_names.txt

local/get_transcriptions.sh yaounde

local/get_utt2text.pl $dir/yaounde_trans_unsorted.txt > $dir/yaounde_utt2text_unsorted.txt

local/get_utt2spk.sh $data yaounde

local/get_spk2utt.sh $data yaounde

local/get_utt2wav_filename.pl $data $dir/yaounde_speaker_names.txt > $dir/yaounde_wav_unsorted.scp

exit
mkdir -p data/train

local/sort_transcriptions.sh train

