#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#                 Korbinian Riedhammer

# Create denominator lattices for MMI/MPE training.
# Creates its output in $dir/lat.*.gz

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.1
max_active=5000
transform_dir=
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/make_tandem_denlats.sh [options] <data1-dir> <data2-dir> <lang-dir> <src-dir> <exp-dir>"
   echo "  e.g.: steps/make_tandem_denlats.sh {mfcc,bottleneck}/data/train data/lang exp/tri1 exp/tri1_denlats"
   echo "Works for (delta|lda) features, and (with --transform-dir option) such features"
   echo " plus transforms."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   echo "  --transform-dir <transform-dir>   # directory to find fMLLR transforms."
   exit 1;
fi

data1=$1
data2=$2
lang=$3
srcdir=$4
dir=$5

mkdir -p $dir/log
echo $nj > $dir/num_jobs

sdata1=$data1/split$nj
sdata2=$data2/split$nj
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir

cp -r $lang $dir/

# Compute grammar FST which corresponds to unigram decoding graph.

cat $data1/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  utils/make_unigram_grammar.pl | fstcompile > $dir/lang/G.fst \
   || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.


if [ -s $dir/dengraph/HCLG.fst ]; then
   echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  utils/mkgraph.sh $dir/lang $srcdir $dir/dengraph || exit 1;
fi


## Set up features.
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $srcdir/normft2 2>/dev/null`

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

case $feat_type in
  delta) 
  	echo "$0: feature type is $feat_type"
  	;;
  lda) 
  	echo "$0: feature type is $feat_type"
    cp $srcdir/{lda,final}.mat $dir/   
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# set up feature stream 1;  this are usually spectral features, so we will add
# deltas or splice them
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- |"

if [ "$feat_type" == "delta" ]; then
  feats1="$feats1 add-deltas ark:- ark:- |"
elif [ "$feat_type" == "lda" ]; then
  feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
fi

# set up feature stream 2;  this are usually bottleneck or posterior features, 
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  feats="$feats transform-feats $dir/final.mat ark:- ark:- |"
fi

# splicing/normalization options
cp $srcdir/{splice_opts,normft2,tandem} $dir 2>/dev/null


if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "$0: using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne "$nj" ] \
    && echo "$0: mismatch in number of jobs with $transform_dir" && exit 1;
  [ -f $srcdir/final.mat ] && ! cmp $transform_dir/final.mat $srcdir/final.mat && \
     echo "$0: LDA transforms differ between $srcdir and $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
else
  if [ -f $srcdir/final.alimdl ]; then
    echo "$0: you seem to have a SAT system but you did not supply the --transform-dir option.";
    exit 1;
  fi
fi


# if this job is interrupted by the user, we want any background jobs to be
# killed too.
cleanup() {
  local pids=$(jobs -pr)
  [ -n "$pids" ] && kill $pids
}
trap "cleanup" INT QUIT TERM EXIT


if [ $sub_split -eq 1 ]; then 
  $cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
   gmm-latgen-faster --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
     $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
else
  # each job from 1 to $nj is split into multiple pieces (sub-split), and we aim
  # to have at most two jobs running at each time.  The idea is that if we have stragglers
  # from one job, we can be processing another one at the same time.
  rm $dir/.error 2>/dev/null

  prev_pid=
  for n in `seq $[nj+1]`; do
    if [ $n -gt $nj ]; then
      this_pid=
    elif [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $srcdir/final.mdl ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
      this_pid=
    else
      ssdata1=$data1/split$nj/$n/split$sub_split;
      if [ ! -d $ssdata1 ] || [ $ssdata1 -ot $sdata1/$n/feats.scp ]; then
        split_data.sh --per-utt $sdata1/$n $sub_split || exit 1;
      fi
      ssdata2=$data2/split$nj/$n/split$sub_split;
      if [ ! -d $ssdata2 ] || [ $ssdata2 -ot $sdata2/$n/feats.scp ]; then
        split_data.sh --per-utt $sdata2/$n $sub_split || exit 1;
      fi
      mkdir -p $dir/log/$n
      mkdir -p $dir/part
      feats_subset=`echo $feats | sed "s/trans.JOB/trans.$n/g" | sed s:JOB/:$n/split$sub_split/JOB/:g`

      $cmd $parallel_opts JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        gmm-latgen-faster --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
        --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
          $dir/dengraph/HCLG.fst "$feats_subset" "ark:|gzip -c >$dir/lat.$n.JOB.gz" || touch .error &
      this_pid=$!
    fi
    if [ ! -z "$prev_pid" ]; then  # Wait for the previous job; merge the previous set of lattices.
      wait $prev_pid
      [ -f $dir/.error ] && echo "$0: error generating denominator lattices" && exit 1;
      rm $dir/.merge_error 2>/dev/null
      echo Merging archives for data subset $prev_n
      for k in `seq $sub_split`; do
        gunzip -c $dir/lat.$prev_n.$k.gz || touch $dir/.merge_error;
      done | gzip -c > $dir/lat.$prev_n.gz || touch $dir/.merge_error;
      [ -f $dir/.merge_error ] && echo "$0: Merging lattices for subset $prev_n failed (or maybe some other error)" && exit 1;
      rm $dir/lat.$prev_n.*.gz
      touch $dir/.done.$prev_n
    fi
    prev_n=$n
    prev_pid=$this_pid
  done
fi


echo "$0: done generating denominator lattices."
