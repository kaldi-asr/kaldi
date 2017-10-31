#!/bin/bash
# Copyright      2012  Brno University of Technology (Author: Karel Vesely)
#           2013-2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments using DNN.  This takes as input a directory
# prepared as for online-nnet2 decoding (e.g. by
# steps/online/nnet2/prepare_online_decoding.sh), and it computes the features
# directly from the wav.scp instead of relying on features dumped on disk;
# this avoids the hassle of having to dump suitably matched features.


# Begin configuration section.  
nj=4
cmd=run.pl
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
iter=final
use_gpu=no

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: $0 <data-dir> <lang-dir> <src-dir> <align-dir>"
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
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;


for f in $srcdir/tree $srcdir/${iter}.mdl $data/wav.scp $lang/L.fst \
      $srcdir/conf/online_nnet2_decoding.conf; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;
cp $srcdir/{tree,${iter}.mdl} $dir || exit 1;

grep -v '^--endpoint' $srcdir/conf/online_nnet2_decoding.conf >$dir/feature.conf || exit 1;


if [ -f $data/segments ]; then
  # note: in the feature extraction, because the program online2-wav-dump-features is sensitive to the
  # previous utterances within a speaker, we do the filtering after extracting the features.
  echo "$0 [info]: segments file exists: using that."
  feats="ark,s,cs:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- | online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt ark,s,cs:- ark:- |"
else
  echo "$0 [info]: no segments file exists, using wav.scp."
  feats="ark,s,cs:online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt scp:$sdata/JOB/wav.scp ark:- |"
fi

echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";

$cmd JOB=1:$nj $dir/log/align.JOB.log \
  compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $srcdir/${iter}.mdl  $lang/L.fst "$tra" ark:- \| \
  nnet-align-compiled $scale_opts --use-gpu=$use_gpu --beam=$beam --retry-beam=$retry_beam \
    $srcdir/${iter}.mdl ark:- "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

echo "$0: done aligning data."

