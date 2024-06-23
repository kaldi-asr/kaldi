#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2020  Hossein Hadian
# Apache 2.0

# This is similar to align_fmllr_lats.sh except it works with non-SAT
# systems e.g. tri1 or tri2

# Computes training alignments using a model with delta or
# LDA+MLLT features.

# If you supply the "--use-graphs true" option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.


# Begin configuration section.
nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
#scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
acoustic_scale=0.1
beam=10
retry_beam=40
retry_beam=40
final_beam=20  # For the lattice-generation phase there is no retry-beam.  This
               # is a limitation of gmm-latgen-faster.  We just use an
               # intermediate beam.  We'll lose a little data and it will be
               # slightly slower.  (however, the min-active of 200 that
               # gmm-latgen-faster defaults to may help.)
careful=false
boost_silence=1.0 # Factor by which to boost silence during alignment.
stage=0
generate_ali_from_lats=false # If true, alingments generated from lattices.
max_active=7000
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_si.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_si.sh data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4


for f in $data/text $lang/oov.int $srcdir/tree $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
cp $srcdir/delta_opts $dir 2>/dev/null

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;



if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $srcdir/full.mat $dir
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac


if [ $stage -le 0 ]; then
  echo "$0: compiling training graphs"
  tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $scale_opts $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi


echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/final.mdl - |"

if [ $stage -le 2 ]; then
  # Warning: gmm-latgen-faster doesn't support a retry-beam so you may get more
  # alignment errors (however, it does have a default min-active=200 so this
  # will tend to reduce alignment errors).
  # --allow_partial=false makes sure we reach the end of the decoding graph.
  # --word-determinize=false makes sure we retain the alternative pronunciations of
  #   words (including alternatives regarding optional silences).
  #  --lattice-beam=$beam keeps all the alternatives that were within the beam,
  #    it means we do no pruning of the lattice (lattices from a training transcription
  #    will be small anyway).
  echo "$0: generating lattices containing alternate pronunciations."
  $cmd JOB=1:$nj $dir/log/generate_lattices.JOB.log \
    gmm-latgen-faster --max-active=$max_active --acoustic-scale=$acoustic_scale --beam=$final_beam \
        --lattice-beam=$final_beam --allow-partial=false --word-determinize=false \
      "$mdl" "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if [ $stage -le 3 ] && $generate_ali_from_lats; then
  # If generate_alignments is true, ali.*.gz is generated in lats dir
  $cmd JOB=1:$nj $dir/log/generate_alignments.JOB.log \
    lattice-best-path --acoustic-scale=$acoustic_scale "ark:gunzip -c $dir/lat.JOB.gz |" \
    ark:/dev/null "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir

echo "$0: done aligning data."
