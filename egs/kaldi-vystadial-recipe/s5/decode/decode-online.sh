#!/bin/bash
# FIXME in general in bad shape
#     1) WER is wrong

# Copyright 2013 Ondrej Platek, based on Vassil Panayotov script
# Apache 2.0

# Set the paths to the binaries and scripts needed
fwd=`dirname $0`
KALDI_ROOT=$fwd/../../../..
export PATH=$fwd/../../s5/utils/:$KALDI_ROOT/src/onlinebin:$KALDI_ROOT/src/bin:$PATH

# Change this to "tri2a" if you like to test using a ML-trained model
ac_model_type=tri2a
exp="$fwd/../Results/expc0bcaa8acd2732dce7c25c27b945d566d80ca7a6"
data="$fwd/../data_voip_en/test"

# Alignments and decoding results are saved in this directory(simulated decoding only)
decode_dir="$fwd/../exp-decode"

# Change this to "live" either here or using command line switch like:
# --test-mode live
test_mode="simulated"

. parse_options.sh

ac_model="$exp/$ac_model_type"
trans_matrix=""


if [ ! -d $ac_model ]; then
    echo "The directory for AC model does not exist: $ac_model "
    exit 1
fi

if [ -s $ac_model/matrix ]; then
    trans_matrix=$ac_model/matrix  # lda matrix
fi

case $test_mode in
    live)
        echo
        echo -e "  LIVE DEMO MODE - you can use a microphone and say something\n"
        echo "Using model in $ac_model directory"
        echo 
        online-gmm-decode-faster --rt-min=0.5 --rt-max=0.7 --max-active=4000 \
           --beam=12.0 --acoustic-scale=0.0769 $ac_model/final.mdl $ac_model/graph/HCLG.fst \
           $ac_model/graph/words.txt '1:2:3:4:5' $trans_matrix;;
    
    simulated)
        echo
        echo -e "  SIMULATED ONLINE DECODING - pre-recorded audio is used\n"
        echo "Test file are from directory $data"
        echo "Using model in $ac_model directory"
        echo 
        ;;
    
    *)
        echo "Invalid test mode! Should be either \"live\" or \"simulated\"!";
        exit 1;;
esac

# Estimate the error rate for the simulated decoding
if [ $test_mode == "simulated" ]; then
    mkdir -p $decode_dir
    # Create new input.scp file
    rm -f $decode_dir/input.scp
    for f in "$data"/*.wav; do
        bf=`basename $f`
        bf=${bf%.wav}
        echo $bf $f >> $decode_dir/input.scp
    done
    # Decode
    online-wav-gmm-decode-faster --verbose=1 --rt-min=0.8 --rt-max=0.85\
        --max-active=4000 --beam=12.0 --acoustic-scale=0.0769 \
        scp:$decode_dir/input.scp $ac_model/final.mdl $ac_model/graph/HCLG.fst \
        $ac_model/graph/words.txt '1:2:3:4:5' ark,t:$decode_dir/trans.txt \
        ark,t:$decode_dir/ali.txt $trans_matrix

    # Create new ref.txt file 
    rm -f "$decode_dir/ref.txt"
    cat $decode_dir/input.scp | tr -s ' ' | cut -d ' ' -f 2- |\
    while read wav_file ; do
        # Convert the reference transcripts from symbols to word IDs
        symbols=`sym2int.pl $ac_model/graph/words.txt < "$wav_file.trn"` 
        name=`basename "$wav_file"`
        name=${name%.wav}
        echo "$name $symbols" >> $decode_dir/ref.txt
    done
    
    # Compact the hypotheses belonging to the same test utterance
    cat $decode_dir/trans.txt | tr -s ' ' | sed -r 's:_[0-9]+-[0-9]+\>::' |\
        gawk '{key=$1; $1=""; arr[key]=arr[key] " " $0; } END { for (k in arr) { print k " " arr[k]} }' > $decode_dir/hyp.txt

    # Finally compute WER
    compute-wer --mode=all --verbose=100 ark,t:$decode_dir/ref.txt ark,t:$decode_dir/hyp.txt
fi
