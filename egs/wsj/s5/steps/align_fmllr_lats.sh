#!/usr/bin/env bash
#
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Version of align_fmllr.sh that generates lattices (lat.*.gz) with
# alignments of alternative pronunciations in them.  Mainly intended
# as a precursor to LF-MMI/chain training for now.

# Begin configuration section.
stage=0
nj=4
cmd=run.pl
# Begin configuration.
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
acoustic_scale=0.1
beam=10
retry_beam=40
final_beam=20  # For the lattice-generation phase there is no retry-beam.  This
               # is a limitation of gmm-latgen-faster.  We just use an
               # intermediate beam.  We'll lose a little data and it will be
               # slightly slower.  (however, the min-active of 200 that
               # gmm-latgen-faster defaults to may help.)
boost_silence=1.0 # factor by which to boost silence during alignment.
fmllr_update_type=full
generate_ali_from_lats=false # If true, alingments generated from lattices.
max_active=7000
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_fmllr_lats.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_fmllr_lats.sh data/train data/lang exp/tri1 exp/tri1_lats"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --fmllr-update-type (full|diag|offset|none)      # default full."
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
cp $srcdir/final.alimdl $dir 2>/dev/null
cp $srcdir/final.occs $dir;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
cp $srcdir/delta_opts $dir 2>/dev/null

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir
    cp $srcdir/full.mat $dir 2>/dev/null
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

## Set up model and alignment model.
mdl=$srcdir/final.mdl
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl
else
  alimdl=$srcdir/final.mdl
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;
alimdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $alimdl - |"
mdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $mdl - |"


## because gmm-latgen-faster doesn't support adding the transition-probs to the
## graph itself, we need to bake them into the compiled graphs.  This means we can't reuse previously compiled graphs,
## because the other scripts write them without transition probs.
if [ $stage -le 0 ]; then
  echo "$0: compiling training graphs"
  tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $scale_opts $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi


if [ $stage -le 1 ]; then
  # Note: we need to set --transition-scale=0.0 --self-loop-scale=0.0 because,
  # as explained above, we compiled the transition probs into the training
  # graphs.
  echo "$0: aligning data in $data using $alimdl and speaker-independent features."
  $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
    gmm-align-compiled --transition-scale=0.0 --self-loop-scale=0.0 --acoustic-scale=$acoustic_scale \
        --beam=$beam --retry-beam=$retry_beam "$alimdl_cmd" \
    "ark:gunzip -c $dir/fsts.JOB.gz|" "$sifeats" "ark:|gzip -c >$dir/pre_ali.JOB.gz" || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing fMLLR transforms"
  if [ "$alimdl" != "$mdl" ]; then
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-post-to-gpost $alimdl "$sifeats" ark:- ark:- \| \
      gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
      ark,s,cs:- ark:$dir/trans.JOB || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
      ark,s,cs:- ark:$dir/trans.JOB || exit 1;
  fi
fi

feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"

if [ $stage -le 3 ]; then
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
      "$mdl_cmd" "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if [ $stage -le 4 ] && $generate_ali_from_lats; then
  # If generate_alignments is true, ali.*.gz is generated in lats dir
  $cmd JOB=1:$nj $dir/log/generate_alignments.JOB.log \
    lattice-best-path --acoustic-scale=$acoustic_scale "ark:gunzip -c $dir/lat.JOB.gz |" \
    ark:/dev/null "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

rm $dir/pre_ali.*.gz 2>/dev/null || true

echo "$0: done generating lattices from training transcripts."

utils/summarize_warnings.pl $dir/log

exit 0;
