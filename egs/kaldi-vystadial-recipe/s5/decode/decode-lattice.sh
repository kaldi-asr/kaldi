#!/bin/bash
# -*- coding: utf-8 -*-
# Author:   Ondrej Platek,2013, code is without any warranty!
# Created:  10:48:02 09/04/2013
# Modified: 16:52:59 10/04/2013

# Set the paths to the binaries and scripts needed
fwd=`dirname $0`
KALDI_ROOT=$fwd/../../../..
export PATH="$PATH":$fwd/../steps/:$fwd/../utils/:$KALDI_ROOT/src/onlinebin:$KALDI_ROOT/src/bin

# Change this to "tri2a" if you like to test using a ML-trained model
ac_model_type=tri2a
exp="$fwd/../Results/expc0bcaa8acd2732dce7c25c27b945d566d80ca7a6"
data="$fwd/../data_voip_en1/test"

# Alignments and decoding results are saved in this directory(simulated decoding only)
decode_dir="$fwd/../exp-decode-lat"

# Change this to "live" either here or using command line switch like:
# --test-mode live # NOT SUPPORTED YET
test_mode="simulated"

# decoding parameters
cmd=run.pl
nj=1   # we do not do data_split as in steps/decode.sh
max_active=7000
beam=13.0
latbeam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
lmwt=9 # TODO setup according experiments
feat_type='delta'

. $fwd/../path.sh; # source the path.
. parse_options.sh || exit 1;


ac_model="$exp/$ac_model_type"

if [ ! -d $ac_model ]; then
    echo "The directory for AC model does not exist: $ac_model "
    exit 1
fi

case $test_mode in
    live)
        echo
        echo "CURRENTLY NOT SUPPORTED!"
        echo -e "  LIVE DEMO MODE - you can use a microphone and say something\n"
        echo "Using model in $ac_model directory"
        echo "CURRENTLY NOT SUPPORTED!"
        echo 
        exit 1;;
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
    # Resets file - do not append 
    rm -f $decode_dir/wav.scp "$decode_dir/ref.txt" "$decode_dir/utt2spk"
    for f in "$data"/*.wav; do
        name=`basename $f`
        echo "$name $f" >> $decode_dir/wav.scp
        echo "$name $name" >> $decode_dir/utt2spk
        # symbols=`sym2int.pl $ac_model/graph/words.txt < "${f}.trn"` 
        symbols=`cat "${f}.trn"` 
        echo "$name $symbols" >> $decode_dir/ref.txt
    done

    # in utils creates utt2spk
    utt2spk_to_spk2utt.pl "$decode_dir"/utt2spk > "$decode_dir/spk2utt"  || exit 1
    # # in steps creates feats.scp FIXME creates wrong scp
    mkdir -p $decode_dir/mfcc
    time ( make_mfcc.sh --cmd "$cmd" --nj $nj $decode_dir $decode_dir $decode_dir/mfcc || exit 1 )
    # in steps creates cmvn.scp
    time ( compute_cmvn_stats.sh $decode_dir $decode_dir $decode_dir/mfcc || exit 1 )

    # Decoding: Based on steps/decode.sh and local/score.sh
    case $feat_type in
          delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$decode_dir/utt2spk scp:$decode_dir/cmvn.scp scp:$decode_dir/feats.scp ark:- | add-deltas ark:- ark:- |";;
          # lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
          *) echo "Invalid feature type $feat_type" && exit 1;
    esac

    # TODO How is the gmm-latgen-paralelized? On data -> bad for us!
    # TODO $nj == 1 does it depend on data? IMHO yes (See steps/decode.sh)
    time ( $cmd JOB=1:$nj $decode_dir/decodeLattice.JOB.log \
     gmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
     --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$ac_model/graph/words.txt \
     $ac_model/final.mdl $ac_model/graph/HCLG.fst "$feats" "ark:|gzip -c > $decode_dir/lat.JOB.gz" || exit 1 )

    time ( lattice-best-path --lm-scale=$lmwt --word-symbol-table=$ac_model/graph/words.txt \
        "ark:gunzip -c $decode_dir/lat.*.gz|" ark,t:$decode_dir/trans.txt || exit 1 )

    # Finally compute WER
    cat $decode_dir/trans.txt | \
      utils/int2sym.pl -f 2- $ac_model/graph/words.txt | sed 's:\<UNK\>::g' | \
      compute-wer --text --mode=present \
      ark:$decode_dir/ref.txt  ark,p:- >& $decode_dir/wer || exit 1;

fi
