#!/bin/bash
#
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey)
#           2017       Hossein Hadian
# Apache 2.0

# Begin configuration section.
stage=0
nj=4
cmd=run.pl
# Begin configuration.
scale_opts="--transition-scale=1.0 --self-loop-scale=1.0"
acoustic_scale=1.0
beam=10
final_beam=20  # For the lattice-generation phase there is no retry-beam.  This
               # is a limitation of gmm-latgen-faster.  We just use an
               # intermediate beam.  We'll lose a little data and it will be
               # slightly slower.  (however, the min-active of 200 that
               # gmm-latgen-faster defaults to may help.)
post_decode_acwt=12.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_nnet3_lats.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_nnet3_lats.sh data/train data/lang exp/tri1 exp/tri1_lats"
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
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
sdata=$data/split$nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor)
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
  cp $srcdir/frame_subsampling_factor $dir
fi

model=$srcdir/final.mdl
if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

if [ $stage -le 1 ]; then
  echo "$0: generating lattices containing alternate pronunciations."
  $cmd JOB=1:$nj $dir/log/generate_lattices.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int \
      $scale_opts $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
      nnet3-latgen-faster $frame_subsampling_opt \
        --frames-per-chunk=$frames_per_chunk \
        --extra-left-context=$extra_left_context \
        --extra-right-context=$extra_right_context \
        --extra-left-context-initial=$extra_left_context_initial \
        --extra-right-context-final=$extra_right_context_final \
        --minimize=false --acoustic-scale=$acoustic_scale --beam=$final_beam \
        --lattice-beam=$final_beam --allow-partial=false \
        --word-determinize=false --word-symbol-table=$lang/words.txt \
        $model ark:- "$feats" \
        "$lat_wspecifier" || exit 1;
fi

echo "$0: done generating lattices from training transcripts."

utils/summarize_warnings.pl $dir/log

exit 0;
