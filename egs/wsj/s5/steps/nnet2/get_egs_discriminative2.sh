#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script dumps examples MPE or MMI or state-level minimum bayes risk (sMBR)
# training of neural nets.  Note: for "criterion", smbr > mpe > mmi in terms of
# compatibility of the dumped egs, meaning you can use the egs dumped with
# --criterion smbr for MPE or MMI, and egs dumped with --criterion mpe for MMI
# training.  The discriminative training program itself doesn't enforce this and
# it would let you mix and match them arbitrarily; we area speaking in terms of
# the correctness of the algorithm that splits the lattices into pieces.

# Begin configuration section.
cmd=run.pl
criterion=smbr
drop_frames=false #  option relevant for MMI, affects how we dump examples.
samples_per_iter=400000 # measured in frames, not in "examples"
max_temp_archives=128 # maximum number of temp archives per input job, only
                      # affects the process of generating archives, not the
                      # final result.

stage=0

cleanup=true
transform_dir= # If this is a SAT system, directory for transforms
online_ivector_dir=
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <denlat-dir> <src-model-file> <degs-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet_denlats exp/tri4/final.mdl exp/tri4_mpe/degs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs (probably would be good to add -tc 5 or so if using"
  echo "                                                   # GridEngine (to avoid excessive NFS traffic)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --criterion <criterion|smbr>                     # Training criterion: may be smbr, mmi or mpfe"
  echo "  --online-ivector-dir <dir|"">                    # Directory for online-estimated iVectors, used in the"
  echo "                                                   # online-neural-net setup.  (but you may want to use"
  echo "                                                   # steps/online/nnet2/get_egs_discriminative2.sh instead)"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
src_model=$5
dir=$6


extra_files=
[ ! -z $online_ivector_dir ] && \
  extra_files="$online_ivector_dir/ivector_period $online_ivector_dir/ivector_online.scp"

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/num_jobs $alidir/tree \
         $denlatdir/lat.1.gz $denlatdir/num_jobs $src_model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log $dir/info || exit 1;


nj=$(cat $denlatdir/num_jobs) || exit 1; # $nj is the number of
                                         # splits of the denlats and alignments.


nj_ali=$(cat $alidir/num_jobs) || exit 1;

sdata=$data/split$nj
utils/split_data.sh $data $nj




if [ $nj_ali -eq $nj ]; then
  ali_rspecifier="ark,s,cs:gunzip -c $alidir/ali.JOB.gz |"
else
  ali_rspecifier="scp:$dir/ali.scp"
  if [ $stage -le 1 ]; then
    echo "$0: number of jobs in den-lats versus alignments differ: dumping them as single archive and index."
    all_ids=$(seq -s, $nj_ali)
    copy-int-vector --print-args=false \
      "ark:gunzip -c $alidir/ali.{$all_ids}.gz|" ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
  fi
fi


splice_opts=`cat $alidir/splice_opts 2>/dev/null`
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null
cp $alidir/tree $dir
cp $lang/phones/silence.csl $dir/info/
cp $src_model $dir/final.mdl || exit 1

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period)
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
  echo $ivector_dim >$dir/info/ivector_dim
  # the 'const_dim_opt' allows it to write only one iVector per example,
  # rather than one per time-index... it has to average over
  const_dim_opt="--const-feat-dim=$ivector_dim"
else
  echo 0 > $dir/info/ivector_dim
fi

## We don't support deltas here, only LDA or raw (mainly because deltas are less
## frequently used).
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ] && [ ! -f $transform_dir/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
   ;;
  lda) 
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    cp $alidir/final.mat $dir    
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ -z "$transform_dir" ]; then
  if [ -f $transform_dir/trans.1 ] || [ -f $transform_dir/raw_trans.1 ]; then
    transform_dir=$alidir
  fi
fi

if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)
  
  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && ! cmp $transform_dir/final.mat $alidir/final.mat; then
    echo "$0: LDA transforms differ between $alidir and $transform_dir"
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
if [ ! -z $online_ivector_dir ]; then
  # add iVectors to the features.
  feats="$feats paste-feats --length-tolerance=$ivector_period ark:- 'ark,s,cs:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |' ark:- |"
fi


if [ $stage -le 2 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)

  echo $num_frames > $dir/info/num_frames 

  # Working out total number of archives. Add one on the assumption the
  # num-frames won't divide exactly, and we want to round up.
  num_archives=$[$num_frames/$samples_per_iter + 1]

  # the next few lines relate to how we may temporarily split each input job
  # into fewer than $num_archives pieces, to avoid using an excessive
  # number of filehandles.
  archive_ratio=$[$num_archives/$max_temp_archives+1]
  num_archives_temp=$[$num_archives/$archive_ratio]
  # change $num_archives slightly to make it an exact multiple
  # of $archive_ratio.
  num_archives=$[$num_archives_temp*$archive_ratio]

  echo $num_archives >$dir/info/num_archives || exit 1
  echo $num_archives_temp >$dir/info/num_archives_temp || exit 1
  
  frames_per_archive=$[$num_frames/$num_archives]

  # note, this is the number of frames per archive prior to discarding frames.
  echo $frames_per_archive > $dir/info/frames_per_archive
else
  num_archives=$(cat $dir/info/num_archives) || exit 1;
  num_archives_temp=$(cat $dir/info/num_archives_temp) || exit 1;
  frames_per_archive=$(cat $dir/info/frames_per_archive) || exit 1;
fi

echo "$0: Splitting the data up into $num_archives archives (using $num_archives_temp temporary pieces per input job)"
echo "$0: giving samples-per-iteration of $frames_per_archive (you requested $samples_per_iter)."

# we create these data links regardless of the stage, as there are situations
# where we would want to recreate a data link that had previously been deleted.

if [ -d $dir/storage ]; then
  echo "$0: creating data links for distributed storage of degs"
  # See utils/create_split_dir.pl for how this 'storage' directory is created.
  for x in $(seq $nj); do
    for y in $(seq $num_archives_temp); do
      utils/create_data_link.pl $dir/degs_orig.$x.$y.ark
    done
  done
  for z in $(seq $num_archives); do
    utils/create_data_link.pl $dir/degs.$z.ark
  done
  if [ $num_archives_temp -ne $num_archives ]; then
    for z in $(seq $num_archives); do
      utils/create_data_link.pl $dir/degs_temp.$z.ark
    done
  fi
fi

if [ $stage -le 3 ]; then
  echo "$0: getting initial training examples by splitting lattices"

  degs_list=$(for n in $(seq $num_archives_temp); do echo ark:$dir/degs_orig.JOB.$n.ark; done)

  $cmd JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs-discriminative --criterion=$criterion --drop-frames=$drop_frames \
      "$src_model" "$feats" "$ali_rspecifier" "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz|" ark:- \| \
    nnet-copy-egs-discriminative $const_dim_opt ark:- $degs_list || exit 1;
  sleep 5;  # wait a bit so NFS has time to write files.
fi

if [ $stage -le 4 ]; then
  
  degs_list=$(for n in $(seq $nj); do echo $dir/degs_orig.$n.JOB.ark; done)

  if [ $num_archives -eq $num_archives_temp ]; then
    echo "$0: combining data into final archives and shuffling it"
    
    $cmd JOB=1:$num_archives $dir/log/shuffle.JOB.log \
      cat $degs_list \| nnet-shuffle-egs-discriminative --srand=JOB ark:- \
       ark:$dir/degs.JOB.ark || exit 1;
  else
    echo "$0: combining and re-splitting data into un-shuffled versions of final archives."

    archive_ratio=$[$num_archives/$num_archives_temp]
    ! [ $archive_ratio -gt 1 ] && echo "$0: Bad archive_ratio $archive_ratio" && exit 1;

    # note: the \$[ .. ] won't be evaluated until the job gets executed.  The
    # aim is to write to the archives with the final numbering, 1
    # ... num_archives, which is more than num_archives_temp.  The list with
    # \$[... ] expressions in it computes the set of final indexes for each
    # temporary index.
    degs_list_out=$(for n in $(seq $archive_ratio); do echo "ark:$dir/degs_temp.\$[((JOB-1)*$archive_ratio)+$n].ark"; done)
    # e.g. if dir=foo and archive_ratio=2, we'd have
    # degs_list_out='foo/degs_temp.$[((JOB-1)*2)+1].ark foo/degs_temp.$[((JOB-1)*2)+2].ark'

    $cmd JOB=1:$num_archives_temp $dir/log/resplit.JOB.log \
      cat $degs_list \| nnet-copy-egs-discriminative --srand=JOB ark:- \
      $degs_list_out || exit 1;
  fi
fi

if [ $stage -le 5 ] && [ $num_archives -ne $num_archives_temp ]; then
  echo "$0: shuffling final archives."

  $cmd JOB=1:$num_archives $dir/log/shuffle.JOB.log \
    nnet-shuffle-egs-discriminative --srand=JOB ark:$dir/degs_temp.JOB.ark \
      ark:$dir/degs.JOB.ark || exit 1
fi

if $cleanup; then
  echo "$0: removing temporary archives."
  for x in $(seq $nj); do
    for y in $(seq $num_archives_temp); do
      file=$dir/degs_orig.$x.$y.ark
      [ -L $file ] && rm $(readlink -f $file); rm $file
    done
  done
  if [ $num_archives_temp -ne $num_archives ]; then
    for z in $(seq $num_archives); do
      file=$dir/degs_temp.$z.ark
      [ -L $file ] && rm $(readlink -f $file); rm $file
    done
  fi
fi

echo "$0: Done."
