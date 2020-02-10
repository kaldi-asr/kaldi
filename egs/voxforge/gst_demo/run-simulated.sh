#!/usr/bin/env bash
#
# Copyright 2012 Vassil Panayotov
# Copyright 2013 Tanel Alumae
# Apache 2.0
#
# This script replicates the simulated online decoding demo in 
# ../online_demo, using the GStreamer plugin
# 

KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/../s5/utils/:$KALDI_ROOT/src/bin:$PATH

data_file="online-data"
data_url="http://sourceforge.net/projects/kaldi/files/online-data.tar.bz2"

# Change this to "tri2a" if you like to test using a ML-trained model
ac_model_type=tri2b_mmi

# Alignments and decoding results are saved in this directory
decode_dir="./work"

ac_model=${data_file}/models/$ac_model_type

ac_model=${data_file}/models/$ac_model_type
trans_matrix=""
audio=${data_file}/audio

if [ ! -s $KALDI_ROOT/src/gst-plugin/libgstonlinegmmdecodefaster.so ]; then
    echo "Kaldi Gstreamer plugin libarary $KALDI_ROOT/src/gst-plugin/libgstonlinegmmdecodefaster.so not present, make it first"
    exit 1
fi

if [ ! -s ${data_file}.tar.bz2 ]; then
    echo "Downloading test models and data ..."
    wget -T 10 -t 3 $data_url;
    
    if [ ! -s ${data_file}.tar.bz2 ]; then
        echo "Download of $data_file has failed!"
        exit 1
    fi
fi

if [ ! -d $ac_model ]; then
    echo "Extracting the models and data ..."
    tar xf ${data_file}.tar.bz2
fi

if [ -s $ac_model/matrix ]; then
    trans_matrix=$ac_model/matrix
fi

mkdir -p work

for f in $audio/*.wav; do
    resultfile=$decode_dir/`basename $f .wav`.hyp
    echo "Decoding $f, result goes to $resultfile"
    GST_PLUGIN_PATH=$KALDI_ROOT/src/gst-plugin  gst-launch-1.0 filesrc location=$f \
      ! decodebin ! audioconvert ! audioresample \
      ! onlinegmmdecodefaster rt-min=0.8 rt-max=0.85  max-active=4000 beam=12.0 acoustic-scale=0.0769 \
                               model=$ac_model/model fst=$ac_model/HCLG.fst \
                               word-syms=$ac_model/words.txt silence-phones="1:2:3:4:5" \
                               lda-mat=$trans_matrix \
      ! filesink location=$resultfile
done

# Convert the reference transcripts from symbols to word IDs
sym2int.pl -f 2- $ac_model/words.txt < $audio/trans.txt > $decode_dir/ref.txt

# Convert the hypotheses from symbols to word IDs, also remove the segmentation symbol "<#s>"
for f in $decode_dir/*.hyp; do
    (echo -n `basename $f .hyp`" " ; cat $f) | sed 's/<#s>//g' | sym2int.pl -f 2- $ac_model/words.txt;
done > $decode_dir/hyp.txt

# Finally compute WER
compute-wer --mode=present ark,t:$decode_dir/ref.txt ark,t:$decode_dir/hyp.txt

