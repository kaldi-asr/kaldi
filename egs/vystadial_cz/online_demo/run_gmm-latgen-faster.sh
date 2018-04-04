# $!/bin/bash

# source the settings
. ./path.sh

. utils/parse_options.sh || exit 1

# temporary files
mfccdir=$decode_dir/mfcc
feat_scp=$mfccdir/raw_mfcc.scp

mkdir -p $mfccdir

compute-mfcc-feats  --verbose=0 --config=$MFCC scp:$wav_scp \
  ark,scp:$mfccdir/raw_mfcc.ark,$feat_scp || exit 1;

# # For debugging
# # cgdb -q -x .gdbinit --args \
# compute-mfcc-feats  --verbose=0 --config=$MFCC scp:$wav_scp \
#   ark,t,scp:$mfccdir/raw_mfcc.ark.txt,${feat_scp}.txt || exit 1;
# # For debugging
# add-deltas "scp,s,cs:$feat_scp" "ark,t:$mfccdir/dd_mfcc.ark.txt"

# check the splice-feats use the default left=right=4 context
if [ -z $MAT ] ; then
  # no LDA matrix -> use delta-delta
  feats="ark,s,cs:copy-feats scp:$feat_scp ark:- | add-deltas ark:- ark:- |"
else
  # LDA matrix specified -> using it
  feats="ark,s,cs:copy-feats scp:$feat_scp ark:- | splice-feats ark:- ark:- | transform-feats $MAT ark:- ark:- |"
fi

gmm-latgen-faster --verbose=0 --max-mem=500000000 \
    --beam=$beam --lattice-beam=$latbeam --max-active=$max_active \
    --allow-partial=true --word-symbol-table=$WST \
    $AM $HCLG "$feats" \
    "ark:|gzip -c > $lattice"

lattice-best-path --verbose=0 --lm-scale=15 --word-symbol-table=$WST \
    "ark:gunzip -c $lattice|" ark,t:$gmm_latgen_faster_tra || exit 1;

cat $gmm_latgen_faster_tra | utils/int2sym.pl -f 2- $WST \
    > $gmm_latgen_faster_tra_txt || exit 1

# reference is named based on wav_scp
./build_reference.py $wav_scp $decode_dir
reference=$decode_dir/`basename $wav_scp`.tra

echo; echo "Reference"; echo
cat $reference
echo; echo "Decoded"; echo
cat $gmm_latgen_faster_tra_txt

compute-wer --text --mode=present ark:$reference ark,p:$gmm_latgen_faster_tra_txt

