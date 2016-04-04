#!/bin/bash
# Copyright 2016 Api.ai (Author: Ilya Platonov)
# Apache 2.0

# This script demonstrates kaldi decoding using pretrained model. It will decode list of wav files.
#
# IMPORTANT: wav files must be in 16kHz, 16 bit little-endian format.
#
# This script tries to follow with what other scripts are doing in terms of directory structures and data handling.
# 
# Use ./download-model.sh script to download asr model
# See https://github.com/api-ai/api-ai-english-asr-model for details about a model and how to use it.

. path.sh
MODEL_DIR="exp/api.ai-model"
DATA_DIR="data/test-corpus"

echo "///////"
echo "// IMPORTANT: wav files must be in 16kHz, 16 bit little-endian format."
echo "//////";

for file in final.mdl HCLG.fst words.txt frame_subsampling_factor; do
	if [ ! -f $MODEL_DIR/$file ]; then
		echo "$MODEL_DIR/$file not found, use ./download-model.sh"
		exit 1;
	fi
done;

for app in nnet3-latgen-faster apply-cmvn lattice-scale; do
	command -v $app >/dev/null 2>&1 || { echo >&2 "$app not found, is kaldi compiled?"; exit 1; }
done;

local/create-corpus.sh $DATA_DIR $@ || exit 1;

echo "///////"
echo "// Computing mfcc and cmvn (cmvn is not really used)"
echo "//////";

 steps/make_mfcc.sh --nj 1 --mfcc-config $MODEL_DIR/mfcc.conf \
      --cmd "run.pl" $DATA_DIR exp/make_mfcc exp/mfcc || { echo "Unable to calculate mfcc, ensure 16kHz, 16 bit little-endian wav format or see log"; exit 1; };
    steps/compute_cmvn_stats.sh $DATA_DIR exp/make_mfcc/ exp/mfcc || exit 1;

echo "///////"
echo "// Doing decoding (see log for results)"
echo "//////";
frame_subsampling_factor=$(cat $MODEL_DIR/frame_subsampling_factor)
nnet3-latgen-faster --frame-subsampling-factor=$frame_subsampling_factor --frames-per-chunk=50 --extra-left-context=0 \
 --extra-right-context=0 --extra-left-context-initial=-1 --extra-right-context-final=-1 \
 --minimize=false --max-active=7000 --min-active=200 --beam=15.0 --lattice-beam=8.0 \
 --acoustic-scale=1.0 --allow-partial=true \
 --word-symbol-table=$MODEL_DIR/words.txt $MODEL_DIR/final.mdl $MODEL_DIR//HCLG.fst \
 "ark,s,cs:apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:$DATA_DIR/utt2spk scp:$DATA_DIR/cmvn.scp scp:$DATA_DIR/feats.scp ark:- |" \
 "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:-  >exp/lat.1"