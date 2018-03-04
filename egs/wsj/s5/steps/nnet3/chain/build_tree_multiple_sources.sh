#!/bin/bash
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#           2017  Vimal Manohar
#  Apache 2.0.

# This script is similar to steps/nnet3/chain/build_tree.sh but supports 
# getting statistics from multiple alignment sources.


# Begin configuration section.
stage=-5
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported.
cmd=run.pl
use_fmllr=true  # If true, fmllr transforms will be applied from the alignment directories.
                # Otherwise, no fmllr will be applied even if alignment directory contains trans.*
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
frame_subsampling_factor=1  # frame subsampling factor of output w.r.t. to the input features
tree_stats_opts=
cluster_phones_opts=
repeat_frames=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 5 ]; then
  echo "Usage: steps/nnet3/chain/build_tree_multiple_sources.sh <#leaves> <lang> <data1> <ali-dir1> [<data2> <ali-dir2> ... <data> <ali-dirN>] <exp-dir>"
  echo " e.g.: steps/nnet3/chain/build_tree_multiple_sources.sh 15000 data/lang data/train_sup exp/tri3_ali data/train_unsup exp/tri3/best_path_train_unsup exp/tree_semi"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --repeat-frames <true|false>                     # Only affects alignment conversion at"
  echo "                                                   # the end. If true, generate an "
  echo "                                                   # alignment using the frame-subsampled "
  echo "                                                   # topology that is repeated "
  echo "                                                   # --frame-subsampling-factor times "
  echo "                                                   # and interleaved, to be the same "
  echo "                                                   # length as the original alignment "
  echo "                                                   # (useful for cross-entropy training "
  echo "                                                   # of reduced frame rate systems)."
  exit 1;
fi

numleaves=$1
lang=$2
dir=${@: -1}  # last argument to the script
shift 2;
data_and_alidirs=( $@ )  # read the remaining arguments into an array
unset data_and_alidirs[${#data_and_alidirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=$[${#data_and_alidirs[@]}]  # number of systems to combine

if (( $num_sys % 2 != 0 )); then
  echo "$0: The data and alignment arguments must be an even number of arguments."
  exit 1
fi

num_sys=$((num_sys / 2))

data=$dir/data_tmp
mkdir -p $data

mkdir -p $dir
alidir=`echo ${data_and_alidirs[1]}`

datadirs=()
alidirs=()
for n in `seq 0 $[num_sys-1]`; do
  datadirs[$n]=${data_and_alidirs[$[2*n]]}
  alidirs[$n]=${data_and_alidirs[$[2*n+1]]}
done

utils/combine_data.sh $data ${datadirs[@]} || exit 1

for f in $data/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null` || exit 1
delta_opts=`cat $alidir/delta_opts 2>/dev/null`

mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
cp $alidir/delta_opts $dir 2>/dev/null # delta option.

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

echo $nj >$dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

# Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

echo "$0: feature type is $feat_type"

feats=()
feats_one=()
for n in `seq 0 $[num_sys-1]`; do
  this_nj=$(cat ${alidirs[$n]}/num_jobs) || exit 1
  this_sdata=${datadirs[$n]}/split$this_nj
  [[ -d $this_sdata && ${datadirs[$n]}/feats.scp -ot $this_sdata ]] || split_data.sh ${datadirs[$n]} $this_nj || exit 1;
  ## Set up speaker-independent features.
  case $feat_type in
    delta) feats[$n]="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$this_sdata/JOB/utt2spk scp:$this_sdata/JOB/cmvn.scp scp:$this_sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
      feats_one[$n]="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$this_sdata/1/utt2spk scp:$this_sdata/1/cmvn.scp scp:$this_sdata/1/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
    lda) feats[$n]="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$this_sdata/JOB/utt2spk scp:$this_sdata/JOB/cmvn.scp scp:$this_sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
      feats_one[$n]="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$this_sdata/1/utt2spk scp:$this_sdata/1/cmvn.scp scp:$this_sdata/1/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
      cp $alidir/final.mat $dir
      cp $alidir/full.mat $dir 2>/dev/null
      ;;
    *) echo "$0: invalid feature type $feat_type" && exit 1;
  esac
  
  if $use_fmllr; then
    if [ ! -f ${alidirs[$n]}/trans.1 ]; then
      echo "$0: Could not find fMLLR transforms in ${alidirs[$n]}"
      exit 1
    fi

    echo "$0: Using transforms from ${alidirs[$n]}"
    feats[i]="${feats[i]} transform-feats --utt2spk=ark:$this_sdata/JOB/utt2spk ark,s,cs:${alidirs[$n]}/trans.JOB ark:- ark:- |"
    feats_one[i]="${feats_one[i]} transform-feats --utt2spk=ark:$this_sdata/1/utt2spk ark,s,cs:${alidirs[$n]}/trans.1 ark:- ark:- |"
  fi

  # Do subsampling of feats, if needed
  if [ $frame_subsampling_factor -gt 1 ]; then
    feats[$n]="${feats[$n]} subsample-feats --n=$frame_subsampling_factor ark:- ark:- |"
    feats_one[$n]="${feats_one[$n]} subsample-feats --n=$frame_subsampling_factor ark:- ark:- |"
  fi
done

if [ $stage -le -5 ]; then
  echo "$0: Initializing monophone model (for alignment conversion, in case topology changed)"

  [ ! -f $lang/phones/sets.int ] && exit 1;
  shared_phones_opt="--shared-phones=$lang/phones/sets.int"
  # get feature dimension
  example_feats="`echo ${feats[0]} | sed s/JOB/1/g`";
  if ! feat_dim=$(feat-to-dim "$example_feats" - 2>/dev/null) || [ -z $feat_dim ]; then
    feat-to-dim "$example_feats" - # to see the error message.
    echo "error getting feature dimension"
    exit 1;
  fi

  for n in `seq 0 $[num_sys-1]`; do
    copy-feats "${feats_one[$n]}" ark:-
  done | copy-feats ark:- ark:$dir/tmp.ark
  
  $cmd $dir/log/init_mono.log \
    gmm-init-mono $shared_phones_opt \
      "--train-feats=ark:subset-feats --n=10 ark:$dir/tmp.ark ark:- |" $lang/topo $feat_dim \
    $dir/mono.mdl $dir/mono.tree || exit 1
fi


if [ $stage -le -4 ]; then
  # Get tree stats.

  for n in `seq 0 $[num_sys-1]`; do
    echo "$0: Accumulating tree stats"
    this_data=${datadirs[$n]}
    this_alidir=${alidirs[$n]}
    this_nj=$(cat $this_alidir/num_jobs) || exit 1
    this_frame_subsampling_factor=1
    if [ -f $this_alidir/frame_subsampling_factor ]; then
      this_frame_subsampling_factor=$(cat $this_alidir/frame_subsampling_factor)
    fi

    if (( $frame_subsampling_factor % $this_frame_subsampling_factor != 0 )); then
      echo "$0: frame-subsampling-factor=$frame_subsampling_factor is not "
      echo "divisible by $this_frame_subsampling_factor (that of $this_alidir)"
      exit 1
    fi

    this_frame_subsampling_factor=$((frame_subsampling_factor / this_frame_subsampling_factor))
    $cmd JOB=1:$this_nj $dir/log/acc_tree.$n.JOB.log \
       convert-ali --frame-subsampling-factor=$this_frame_subsampling_factor \
           $this_alidir/final.mdl $dir/mono.mdl $dir/mono.tree "ark:gunzip -c $this_alidir/ali.JOB.gz|" ark:-  \| \
        acc-tree-stats $context_opts $tree_stats_opts --ci-phones=$ciphonelist $dir/mono.mdl \
           "${feats[$n]}" ark:- $dir/$n.JOB.treeacc || exit 1;
    [ "`ls $dir/$n.*.treeacc | wc -w`" -ne "$this_nj" ] && echo "$0: Wrong #tree-accs for data $n $this_data" && exit 1;
  done

  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -3 ] && $train_tree; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  $cmd $dir/log/questions.log \
     cluster-phones $cluster_phones_opts $context_opts $dir/treeacc \
     $lang/phones/sets.int $dir/questions.int || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  $cmd $dir/log/compile_questions.log \
    compile-questions \
      $context_opts $lang/topo $dir/questions.int $dir/questions.qst || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "$0: Initializing the model"
  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;
  grep 'no stats' $dir/log/init_model.log && echo "This is a bad warning.";
  rm $dir/treeacc
fi

if [ $stage -le -1 ]; then
  # Convert the alignments to the new tree.  Note: we likely will not use these
  # converted alignments in the chain system directly, but they could be useful
  # for other purposes.

  for n in `seq 0 $[num_sys-1]`; do
    this_alidir=${alidirs[$n]}
    this_nj=$(cat $this_alidir/num_jobs) || exit 1
    
    this_frame_subsampling_factor=1
    if [ -f $this_alidir/frame_subsampling_factor ]; then
      this_frame_subsampling_factor=$(cat $this_alidir/frame_subsampling_factor)
    fi

    if (( $frame_subsampling_factor % $this_frame_subsampling_factor != 0 )); then
      echo "$0: frame-subsampling-factor=$frame_subsampling_factor is not "
      echo "divisible by $this_frame_subsampling_factor (hat of $this_alidir)"
      exit 1
    fi

    echo "$0: frame-subsampling-factor for $this_alidir is $this_frame_subsampling_factor"

    this_frame_subsampling_factor=$((frame_subsampling_factor / this_frame_subsampling_factor))
    echo "$0: Converting alignments from $this_alidir to use current tree"
    $cmd JOB=1:$this_nj $dir/log/convert.$n.JOB.log \
      convert-ali --repeat-frames=$repeat_frames \
        --frame-subsampling-factor=$this_frame_subsampling_factor \
        $this_alidir/final.mdl $dir/1.mdl $dir/tree "ark:gunzip -c $this_alidir/ali.JOB.gz |" \
        ark,scp:$dir/ali.$n.JOB.ark,$dir/ali.$n.JOB.scp || exit 1

    for i in `seq $this_nj`; do 
      cat $dir/ali.$n.$i.scp 
    done > $dir/ali.$n.scp || exit 1
  done

  for n in `seq 0 $[num_sys-1]`; do
    cat $dir/ali.$n.scp
  done | sort -k1,1 > $dir/ali.scp || exit 1

  utils/split_data.sh $data $nj
  $cmd JOB=1:$nj $dir/log/copy_alignments.JOB.log \
    copy-int-vector "scp:utils/filter_scp.pl $data/split$nj/JOB/utt2spk $dir/ali.scp |" \
    "ark:| gzip -c > $dir/ali.JOB.gz" || exit 1
fi

cp $dir/1.mdl $dir/final.mdl

echo $0: Done building tree
