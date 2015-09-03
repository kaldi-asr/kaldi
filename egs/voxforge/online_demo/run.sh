#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

# Set the paths to the binaries and scripts needed
KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/../s5/utils/:$KALDI_ROOT/src/onlinebin:$KALDI_ROOT/src/bin:$PATH

data_file="online-data"
data_url="http://sourceforge.net/projects/kaldi/files/online-data.tar.bz2"

# Change this to "tri2a" if you like to test using a ML-trained model
ac_model_type=tri2b_mmi

# Alignments and decoding results are saved in this directory(simulated decoding only)
decode_dir="./work"

# Change this to "live" either here or using command line switch like:
# --test-mode live
test_mode="simulated"

. parse_options.sh

ac_model=${data_file}/models/$ac_model_type
trans_matrix=""
audio=${data_file}/audio

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

case $test_mode in
    live)
        echo
        echo -e "  LIVE DEMO MODE - you can use a microphone and say something\n"
        echo "  The (bigram) language model used to build the decoding graph was"
        echo "  estimated on an audio book's text. The text in question is"
        echo "  \"King Solomon's Mines\" (http://www.gutenberg.org/ebooks/2166)."
        echo "  You may want to read some sentences from this book first ..."
        echo 
        online-gmm-decode-faster --rt-min=0.5 --rt-max=0.7 --max-active=4000 \
           --beam=12.0 --acoustic-scale=0.0769 $ac_model/model $ac_model/HCLG.fst \
           $ac_model/words.txt '1:2:3:4:5' $trans_matrix;;
    
    simulated)
        echo
        echo -e "  SIMULATED ONLINE DECODING - pre-recorded audio is used\n"
        echo "  The (bigram) language model used to build the decoding graph was"
        echo "  estimated on an audio book's text. The text in question is"
        echo "  \"King Solomon's Mines\" (http://www.gutenberg.org/ebooks/2166)."
        echo "  The audio chunks to be decoded were taken from the audio book read"
        echo "  by John Nicholson(http://librivox.org/king-solomons-mines-by-haggard/)"
        echo
        echo "  NOTE: Using utterances from the book, on which the LM was estimated"
        echo "        is considered to be \"cheating\" and we are doing this only for"
        echo "        the purposes of the demo."
        echo 
        echo "  You can type \"./run.sh --test-mode live\" to try it using your"
        echo "  own voice!"
        echo
        mkdir -p $decode_dir
        # make an input .scp file
        > $decode_dir/input.scp
        for f in $audio/*.wav; do
            bf=`basename $f`
            bf=${bf%.wav}
            echo $bf $f >> $decode_dir/input.scp
        done
        online-wav-gmm-decode-faster --verbose=1 --rt-min=0.8 --rt-max=0.85\
            --max-active=4000 --beam=12.0 --acoustic-scale=0.0769 \
            scp:$decode_dir/input.scp $ac_model/model $ac_model/HCLG.fst \
            $ac_model/words.txt '1:2:3:4:5' ark,t:$decode_dir/trans.txt \
            ark,t:$decode_dir/ali.txt $trans_matrix;;
    
    *)
        echo "Invalid test mode! Should be either \"live\" or \"simulated\"!";
        exit 1;;
esac

# Estimate the error rate for the simulated decoding
if [ $test_mode == "simulated" ]; then
    # Convert the reference transcripts from symbols to word IDs
    sym2int.pl -f 2- $ac_model/words.txt < $audio/trans.txt > $decode_dir/ref.txt
    
    # Compact the hypotheses belonging to the same test utterance
    cat $decode_dir/trans.txt |\
        sed -e 's/^\(test[0-9]\+\)\([^ ]\+\)\(.*\)/\1 \3/' |\
        gawk '{key=$1; $1=""; arr[key]=arr[key] " " $0; } END { for (k in arr) { print k " " arr[k]} }' > $decode_dir/hyp.txt

   # Finally compute WER
   compute-wer --mode=present ark,t:$decode_dir/ref.txt ark,t:$decode_dir/hyp.txt
fi
