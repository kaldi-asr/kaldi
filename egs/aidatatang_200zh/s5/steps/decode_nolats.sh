#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey)
#                      Vimal Manohar
# Apache 2.0

##Changes
# Vimal Manohar (Jan 2014):
# Added options to boost silence probabilities in the model before
# decoding. This can help in favoring the silence phones when 
# some silence regions are wrongly decoded as speech phones like glottal stops

# Begin configuration section.  
transform_dir=
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
boost_silence=1.0         # Boost silence pdfs in the model by this factor before decoding
silence_phones_list=      # List of silence phones that would be boosted before decoding
stage=0
nj=4
cmd=run.pl
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
write_alignments=false  # The output directory is treated like an alignment directory
write_words=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

[ -z $silence_phones_list ] && boost_silence=1.0

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is.  This version produces just linear output, no lattices"
   echo ""
   echo "e.g.: steps/decode.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --write-alignments <true|false>                  # if true, output ali.*.gz"
   echo "  --write-words <true|false>                       # if true, output words.*.gz"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; 
  else model=$srcdir/$iter.mdl; fi
fi

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode.sh: no such file $f" && exit 1;
done

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode.sh: feature type is $feat_type";

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

utils/lang/check_phones_compatible.sh $graphdir/phones.txt $srcdir/phones.txt || exit 1;

if $write_alignments; then
  # Copy model and options that are generally expected in an alignment 
  # directory.
  cp $graphdir/phones.txt $dir || exit 1;

  cp $srcdir/{tree,final.mdl} $dir || exit 1;
  cp $srcdir/final.alimdl $dir 2>/dev/null
  cp $srcdir/final.occs $dir;
  cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
  cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
  cp $srcdir/delta_opts $dir 2>/dev/null
fi

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    if $write_alignments; then
      cp $srcdir/final.mat $dir
      cp $srcdir/full.mat $dir 2>/dev/null
    fi
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne $nj ] && \
     echo "Mismatch in number of jobs with $transform_dir";
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi

if [ $stage -le 0 ]; then
  if $write_alignments; then
    ali="ark:|gzip -c > $dir/ali.JOB.gz"
  else
    ali="ark:/dev/null"
  fi
  if $write_words; then
    words="ark:|gzip -c > $dir/words.JOB.gz"
  else
    words="ark:/dev/null"
  fi

  [ ! -z "$silence_phones_list" ]  && \
    model="gmm-boost-silence --boost=$boost_silence $silence_phones_list $model - |"

  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $model"; exit 1; }
  fi
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    gmm-decode-faster --max-active=$max_active --beam=$beam  \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    "$model" $graphdir/HCLG.fst "$feats" "$words" "$ali" || exit 1;
fi

exit 0;
