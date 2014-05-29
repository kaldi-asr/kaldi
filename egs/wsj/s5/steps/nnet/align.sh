#!/bin/bash
# Copyright 2012-2013 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Aligns 'data' to sequences of transition-ids using Neural Network based acoustic model.
# Optionally produces alignment in lattice format, this is handy to get word alignment.

# Begin configuration section.  
nj=4
cmd=run.pl
stage=0
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40

align_to_lats=false # optionally produce alignment in lattice format
 lats_decode_opts="--acoustic-scale=0.1 --beam=20 --lattice_beam=10"
 lats_graph_scales="--transition-scale=1.0 --self-loop-scale=0.1"

use_gpu="no" # yes|no|optionaly
# End configuration options.

[ $# -gt 0 ] && echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;

# Select default locations to model files
nnet=$srcdir/final.nnet;
class_frame_counts=$srcdir/ali_train_pdf.counts
feature_transform=$srcdir/final.feature_transform
model=$dir/final.mdl

# Check that files exist
for f in $sdata/1/feats.scp $sdata/1/text $lang/L.fst $nnet $model $feature_transform $class_frame_counts; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done


# PREPARE FEATURE EXTRACTION PIPELINE
# Create the feature stream:
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# Optionally add cmvn
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $sdata/1/cmvn.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
fi
# Optionally add deltas
if [ -f $srcdir/delta_order ]; then
  delta_order=$(cat $srcdir/delta_order)
  feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
fi
# Finally add feature_transform and the MLP
feats="$feats nnet-forward --feature-transform=$feature_transform --no-softmax=true --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet ark:- ark:- |"


echo "$0: aligning data '$data' using nnet/model '$srcdir', putting alignments in '$dir'"
# Map oovs in reference transcription 
tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
# We could just use align-mapped in the next line, but it's less efficient as it compiles the
# training graphs one by one.
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/final.mdl ark:- \
      "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

# Optionally align to lattice format (handy to get word alignment)
if [ "$align_to_lats" == "true" ]; then
  echo "$0: aligning also to lattices '$dir/lat.*.gz'"
  $cmd JOB=1:$nj $dir/log/align_lat.JOB.log \
    compile-train-graphs $lat_graph_scale $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    latgen-faster-mapped $lat_decode_opts --word-symbol-table=$lang/words.txt $dir/final.mdl ark:- \
      "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

echo "$0: done aligning data."
