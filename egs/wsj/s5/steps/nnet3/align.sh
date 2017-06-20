#!/bin/bash
# Copyright 2012  Brno University of Technology (Author: Karel Vesely)
#           2013  Johns Hopkins University (Author: Daniel Povey)
#           2015  Vijayaditya Peddinti
#           2016  Vimal Manohar
# Apache 2.0

# Computes training alignments using nnet3 DNN

# Begin configuration section.
nj=4
cmd=run.pl
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
transform_dir=
iter=final
use_gpu=true
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
feat_type=  # you can set this to force it to use delta features.
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: $0 [--transform-dir <transform-dir>] <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.: $0 data/train data/lang exp/nnet4 exp/nnet4_ali"
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
sdata=$data/split${nj}utt
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
   split_data.sh --per-utt $data $nj || exit 1;

if $use_gpu; then
  queue_opt="--gpu 1"
  gpu_opt="--use-gpu=yes"
else
  queue_opt=""
  gpu_opt="--use-gpu=no"
fi

extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

for f in $srcdir/tree $srcdir/${iter}.mdl $data/feats.scp $lang/L.fst $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

cp $srcdir/{tree,${iter}.mdl} $dir || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;
## Set up features.  Note: these are different from the normal features
## because we have one rspecifier that has the features for the entire
## training set, not separate ones for each batch.
if [ -z "$feat_type" ]; then
  if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"

cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
   ;;
  lda)
    splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
    cp $srcdir/splice_opts $dir 2>/dev/null
    cp $srcdir/final.mat $dir || exit 1;
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)

  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && ! cmp $transform_dir/final.mat $srcdir/final.mat; then
    echo "$0: LDA transforms differ between $srcdir and $transform_dir"
    exit 1;
  fi
  if [ ! -f $transform_dir/$trans.1 ]; then
    echo "$0: expected $transform_dir/$trans.1 to exist (--transform-dir option)"
    exit 1;
  fi
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/$trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/$trans.ark,$dir/$trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/$trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/$trans.JOB ark:- ark:- |"
  fi
fi

ivector_opts=
if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor)
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
  cp $srcdir/frame_subsampling_factor $dir
  if [ "$frame_subsampling_factor" -gt 1 ] && \
     [ "$scale_opts" == "--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" ]; then
    echo "$0: frame-subsampling-factor is not 1 (so likely a chain system),"
    echo "...  but the scale opts are the defaults.  You probably want"
    echo "--scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0'"
    sleep 1
  fi
fi


$cmd $queue_opt JOB=1:$nj $dir/log/align.JOB.log \
  compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $srcdir/${iter}.mdl  $lang/L.fst "$tra" ark:- \| \
  nnet3-align-compiled $scale_opts $ivector_opts $frame_subsampling_opt \
  --frames-per-chunk=$frames_per_chunk \
  --extra-left-context=$extra_left_context \
  --extra-right-context=$extra_right_context \
  --extra-left-context-initial=$extra_left_context_initial \
  --extra-right-context-final=$extra_right_context_final \
  $gpu_opt --beam=$beam --retry-beam=$retry_beam \
  $srcdir/${iter}.mdl ark:- "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir

echo "$0: done aligning data."
