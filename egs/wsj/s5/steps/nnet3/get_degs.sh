#!/bin/bash

# Copyright 2012-2016   Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright 2014-2015   Vimal Manohar

# Decodes denlats and dumps egs for discriminative training, in one script
# (avoids writing the non-compact lattices to disk, which can use a lot of disk
# space).


# Begin configuration section.
cmd=run.pl
max_copy_jobs=5  # Limit disk I/O

# feature options
feat_type=raw     # set it to 'lda' to use LDA features.
transform_dir= # If this is a SAT system, directory for transforms
online_ivector_dir=

# example splitting and context options
frames_per_eg=150 # number of frames of labels per example.
                  # Note: may in general be a comma-separated string of alternative
                  # durations; the first one (the principal num-frames) is preferred.
frames_overlap_per_eg=30 # number of supervised frames of overlap that we aim for per eg.
                  # can be useful to avoid wasted data if you're using --left-deriv-truncate
                  # and --right-deriv-truncate.
looped=false       # Set to true to enable looped decoding [can
                   # be a bit faster, for forward-recurrent models like LSTMs.]

# .. these context options also affect decoding.
extra_left_context=0    # amount of left-context per eg, past what is required by the model
                        # (only useful for recurrent networks like LSTMs/BLSTMs)
extra_right_context=0   # amount of right-context per eg, past what is required by the model
                        # (only useful for backwards-recurrent networks like BLSTMs)
extra_left_context_initial=-1    # if >= 0, the --extra-left-context to use at
                                 # the start of utterances.  Recommend 0 if you
                                 # used 0 for the baseline DNN training; if <0,
                                 # defaults to same as extra_left_context
extra_right_context_final=-1     # if >= 0, the --extra-right-context to use at
                                 # the end of utterances.  Recommend 0 if you
                                 # used 0 for the baseline DNN training; if <0,
                                 # defaults to same as extra_left_context

compress=true   # set this to false to disable lossy compression of features
                # dumped with egs (e.g. if you want to see whether results are
                # affected).

num_utts_subset=80     # number of utterances in validation and training
                       # subsets used for diagnostics.
num_egs_subset=800     # number of egs (maximum) for the validation and training
                       # subsets used for diagnostics.
frames_per_iter=1000000 # each iteration of training, see this many frames
                        # per job.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.
cleanup=true

stage=0
nj=200

# By default this script uses final.mdl in <srcdir>, this configures it.
iter=final


# decoding-graph option
self_loop_scale=0.1  # for decoding graph.. should be 1.0 for chain models.

# options relating to decoding.
frames_per_chunk_decoding=150
beam=13.0
lattice_beam=7.0
acwt=0.1
max_active=5000
min_active=200
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
num_threads=1

# affects whether we invoke lattice-determinize-non-compact after decoding
# discriminative-get-supervision.
determinize_before_split=true


# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0 [opts] <data> <lang> <src-dir> <ali-dir> <degs-dir>"
  echo " e.g.: $0 data/train data/lang exp/nnet3/tdnn_a exp/nnet3/tdnn_a_ali exp/nnet3/tdnn_a_degs"
  echo ""
  echo "For options, see top of script file.  Standard options:"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs (probably would be good to add --max-jobs-run 5 or so if using"
  echo "                                                   # GridEngine (to avoid excessive NFS traffic)."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --online-ivector-dir <dir|"">                    # Directory for online-estimated iVectors, used in the"
  echo "                                                   # online-neural-net setup."
  echo "  --nj <nj|200>                                    # number of jobs to submit to the queue."
  echo "  --num-threads <n|1>                              # number of threads per decoding job"
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4
dir=$5


extra_files=
[ ! -z $online_ivector_dir ] && \
  extra_files="$extra_files $online_ivector_dir/ivector_period $online_ivector_dir/ivector_online.scp"
[ "$feat_type" = "lda" ] && \
  extra_files="$extra_files $srcdir/final.mat"
[ ! -z $transform_dir ] && \
  extra_files="$extra_files $transform_dir/trans.1 $transform_dir/num_jobs"

# Check some files.
for f in $data/feats.scp $lang/L.fst $lang/phones/silence.csl $srcdir/${iter}.mdl $srcdir/tree \
      $srcdir/cmvn_opts $alidir/ali.1.gz $alidir/num_jobs $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log $dir/info || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;



utils/split_data.sh --per-utt $data $nj
sdata=$data/split${nj}utt


## Set up features.
if [ -z "$feat_type" ]; then
  if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"


cmvn_opts=$(cat $srcdir/cmvn_opts) || exit 1

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
   ;;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

cp $srcdir/{splice_opts,cmvn_opts} $dir 2>/dev/null || true

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


## set iVector options
if [ ! -z "$online_ivector_dir" ]; then
  online_ivector_period=$(cat $online_ivector_dir/ivector_period)
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$online_ivector_period"
fi

## set frame-subsampling-factor option and copy file
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor) || exit 1
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
  cp $srcdir/frame_subsampling_factor $dir
  if [ $frame_subsampling_factor -ne 1 ] && [ "$self_loop_scale" == "0.1" ]; then
    echo "$0: warning: frame_subsampling_factor is not 1 (so likely a chain system),"
    echo "...  but self-loop-scale is 0.1.  Make sure this is not a mistake."
    sleep 1
  fi
else
  frame_subsampling_factor=1
fi

if [ "$self_loop_scale" == "1.0" ] && [ "$acwt" == 0.1 ]; then
  echo "$0: warning: you set --self-loop-scale=1.0 (so likely a chain system)",
  echo " ... but the acwt is still 0.1 (you probably want --acwt 1.0)"
  sleep 1
fi

## Make the decoding graph.
if [ $stage -le 0 ]; then
  new_lang="$dir/"$(basename "$lang")
  rm -r $new_lang 2>/dev/null
  cp -rH $lang $dir
  echo "$0: Making unigram grammar FST in $new_lang"
  oov=$(cat data/lang/oov.txt)
  cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
   awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
    utils/make_unigram_grammar.pl | fstcompile | fstarcsort --sort_type=ilabel > $new_lang/G.fst \
    || exit 1;

  utils/mkgraph.sh --self-loop-scale $self_loop_scale $new_lang $srcdir $dir/dengraph || exit 1;
fi

# copy alignments into ark,scp format which allows us to use different num-jobs
# from the alignment, and is also convenient for getting priors.
if [ $stage -le 1 ]; then
  echo "$0: Copying input alignments"
  nj_ali=$(cat $alidir/num_jobs)
  alis=$(for n in $(seq $nj_ali); do echo -n "$alidir/ali.$n.gz "; done)
  $cmd $dir/log/copy_alignments.log \
     copy-int-vector "ark:gunzip -c $alis|" \
     ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi

[ -f $dir/ali.scp ] || { echo "$0: expected $dir/ali.scp to exist"; exit 1; }

if [ $stage -le 2 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames
  echo "$0: working out feature dim"
  feats_one="$(echo $feats | sed s:JOB:1:g)"
  if feat_dim=$(feat-to-dim "$feats_one" - 2>/dev/null); then
    echo $feat_dim > $dir/info/feat_dim
  else # run without stderr redirection to show the error.
    feat-to-dim "$feats_one" -; exit 1
  fi
else
  num_frames=$(cat $dir/info/num_frames)
fi
if ! [ "$num_frames" -gt 0 ]; then
  echo "$0: bad num-frames=$num_frames"; exit 1
fi

# copy the model to the degs directory.
cp $srcdir/${iter}.mdl $dir/final.mdl || exit 1

# Create some info in $dir/info

# Work out total number of archives. Add one on the assumption the
# num-frames won't divide exactly, and we want to round up.
num_archives=$[num_frames/frames_per_iter+1]

echo $num_archives >$dir/info/num_archives
echo $frame_subsampling_factor >$dir/info/frame_subsampling_factor
cp $lang/phones/silence.csl $dir/info/

# the first field in frames_per_eg (which is a comma-separated list of numbers)
# is the 'principal' frames-per-eg, and for purposes of working out the number
# of archives we assume that this will be the average number of frames per eg.
frames_per_eg_principal=$(echo $frames_per_eg | cut -d, -f1)


# read 'mof' as max_open_filehandles.
# When splitting up the scp files, we don't want to have to hold too many
# files open at once.  If the number of archives we have to write exceeds
# 256 (or less if unlimit -n is smaller), we split in two stages.
mof=$(ulimit -n) || exit 1
# the next step helps work around inconsistency between different machines on a
# cluster.  It's unlikely that the allowed number of open filehandles would ever
# be less than 256.
if [ $mof -gt 256 ]; then mof=256; fi
# allocate mof minus 3 for the max allowed outputs, because of
# stdin,stderr,stdout.  this will normally come to 253.  We'll do a two-stage
# splitting if the needed number of scp files is larger than this.
num_groups=$[(num_archives+(mof-3)-1)/(mof-3)]
group_size=$[(num_archives+num_groups-1)/num_groups]
if [ $num_groups -gt 1 ]; then
  new_num_archives=$[group_size*num_groups]
  [ $new_num_archives -ne $num_archives ] && \
    echo "$0: rounding up num-archives from $num_archives to $new_num_archives for easier splitting"
  num_archives=$new_num_archives
  echo $new_num_archives >$dir/info/num_archives
fi


if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/degs.$x.ark; done)
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/degs.$x.scp; done)
  utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/degs_orig.$y.ark; done)
  utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/degs_orig.$y.scp; done)
  utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/degs_orig_filtered.$y.scp; done)
fi


extra_context_opts="--extra-left-context=$extra_left_context --extra-right-context=$extra_right_context --extra-left-context-initial=$extra_left_context_initial --extra-right-context-final=$extra_right_context_final"

# work out absolute context opts, --left-context and so on [need model context]
model_left_context=$(nnet3-am-info $srcdir/${iter}.mdl | grep "^left-context:" | awk '{print $2}')
model_right_context=$(nnet3-am-info $srcdir/${iter}.mdl | grep "^right-context:" | awk '{print $2}')
left_context=$[model_left_context+extra_left_context+frame_subsampling_factor/2]
right_context=$[model_right_context+extra_right_context+frame_subsampling_factor/2]
context_opts="--left-context=$left_context --right-context=$right_context"
if [ $extra_left_context_initial -ge 0 ]; then
  left_context_initial=$[model_left_context+extra_left_context_initial+frame_subsampling_factor/2]
  context_opts="$context_opts --left-context-initial=$left_context_initial"
fi
if [ $extra_right_context_final -ge 0 ]; then
  right_context_final=$[model_right_context+extra_right_context_final+frame_subsampling_factor/2]
  context_opts="$context_opts --right-context-final=$right_context_final"
fi

##
if [ $num_threads -eq 1 ]; then
  if $looped; then
    decoder="nnet3-latgen-faster-looped"
    [ $extra_left_context_initial -ge 0 ] && \
      decoder="$decoder --extra-left-context-initial=$extra_left_context_initial"
  else
    decoder="nnet3-latgen-faster $extra_context_opts"
  fi
  threads_cmd_opt=
else
  $looped && { echo "$0: --num-threads must be one if you use looped decoding"; exit 1; }
  threads_cmd_opt="--num-threads $num_threads"
  decoder="nnet3-latgen-faster-parallel --num-threads=$num_threads $extra_context_opts"
  true
fi

# set the command to determinize lattices, if specified.
if $determinize_before_split; then
  lattice_determinize_cmd="lattice-determinize-non-compact --acoustic-scale=$acwt --max-mem=$max_mem --minimize=true --prune=true --beam=$lattice_beam ark:- ark:-"
else
  lattice_determinize_cmd="cat"
fi

if [ $stage -le 3 ]; then
  echo "$0: decoding and dumping egs"
  $cmd $threads_cmd_opt JOB=1:$nj $dir/log/decode_and_get_egs.JOB.log \
     $decoder \
     $ivector_opts $frame_subsampling_opt \
    --frames-per-chunk=$frames_per_chunk_decoding \
    --determinize-lattice=false \
    --max-active=$max_active --min-active=$min_active --beam=$beam \
    --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=false \
    --word-symbol-table=$lang/words.txt $dir/final.mdl  \
    $dir/dengraph/HCLG.fst "$feats" ark:- \| \
    $lattice_determinize_cmd  \| \
    nnet3-discriminative-get-egs --acoustic-scale=$acwt --compress=$compress \
      $frame_subsampling_opt --num-frames=$frames_per_eg \
      --num-frames-overlap=$frames_overlap_per_eg \
      $ivector_opts $context_opts \
      $dir/final.mdl "$feats"  "ark,s,cs:-" \
      "scp:utils/filter_scp.pl $sdata/JOB/utt2spk $dir/ali.scp |" \
      ark,scp:$dir/degs_orig.JOB.ark,$dir/degs_orig.JOB.scp || exit 1
fi


if [ $stage -le 4 ]; then
  echo "$0: getting validation utterances."

  ## Get list of validation utterances.
  awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
   > $dir/valid_uttlist || exit 1;

  if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
    echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
    echo "include all perturbed versions of the same 'real' utterances."
    mv $dir/valid_uttlist $dir/valid_uttlist.tmp
    utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt
    cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
      sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
      awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
    rm $dir/uniq2utt $dir/valid_uttlist.tmp
  fi

  # the following awk statement turns 'foo123' into something like
  # '^foo123-[0-9]\+ ' which is a grep expression that matches the lines in the
  # .scp file that correspond to an utterance in valid_uttlist.
  cat $dir/valid_uttlist | awk '{printf("^%s-[0-9]\\+ \n", $1);}' \
     >$dir/valid_uttlist.regexps || exit 1

  # remove the validation utterances from deg_orig.*.scp to produce
  # degs_orig_filtered.*.scp.
  # note: the '||' true is in case the grep returns nonzero status for
  # some splits, because they were all validation utterances.
  $cmd JOB=1:$nj $dir/log/filter_and_shuffle.JOB.log \
     grep -v -f $dir/valid_uttlist.regexps $dir/degs_orig.JOB.scp '>' \
     $dir/degs_orig_filtered.JOB.scp '||' true || exit 1

  # extract just the validation utterances from deg_orig.*.scp to produce
  # degs_valid.*.scp.
  $cmd JOB=1:$nj $dir/log/extract_validation_egs.JOB.log \
    grep -f $dir/valid_uttlist.regexps $dir/degs_orig.JOB.scp '>' \
    $dir/degs_valid.JOB.scp '||' true || exit 1

  for j in $(seq $nj); do
    cat $dir/degs_valid.$j.scp; rm $dir/degs_valid.$j.scp;
  done | utils/shuffle_list.pl | head -n$num_utts_subset >$dir/valid_diagnostic.scp || exit 1

  [ -s $dir/valid_diagnostic.scp ] || { echo "$0: error getting validation egs"; exit 1; }
fi



# function/pseudo-command to randomly shuffle input lines using a small buffer size
function shuffle {
    perl -e ' use List::Util qw(shuffle); srand(0);
       $bufsz=1000; @A = (); while(<STDIN>) { push @A, $_; if (@A == $bufsz) {
       $n=int(rand()*$bufsz); print $A[$n]; $A[$n] = $A[$bufsz-1]; pop @A; }}
       @A = shuffle(@A); print @A; '
}
# funtion/pseudo-command to put input lines round robin to command line args.
function round_robin {
  perl -e '@F=(); foreach $a (@ARGV) { my $f; open($f, ">$a") || die "opening file $a"; push @F, $f; }
         $N=@F; $N>0||die "No output files"; $n=0;
         while (<STDIN>) { $fh=$F[$n%$N]; $n++; print $fh $_ || die "error printing"; } ' $*
}


if [ $stage -le 5 ]; then
  echo "$0: rearranging scp files"

  if [ $num_groups -eq 1 ]; then
    # output directly to the archive files.
    outputs=$(for n in $(seq $num_archives); do echo $dir/degs.$n.scp; done)
  else
    # output to intermediate 'group' files.
    outputs=$(for g in $(seq $num_groups); do echo $dir/degs_group.$g.scp; done)
  fi

  # We can't use UNIX's split command because of compatibility issues (BSD
  # version very different from GNU version), so we use 'round_robin' which is
  # a bash function that calls an inline perl script.
  for j in $(seq $nj); do cat $dir/degs_orig_filtered.$j.scp; done | \
    shuffle | round_robin $outputs || exit 1

  if [ $num_groups -gt 1 ]; then
    for g in $(seq $num_groups); do
      first=$[1+group_size*(g-1)]
      last=$[group_size*g]
      outputs=$(for n in $(seq $first $last); do echo $dir/degs.$n.scp; done)
      cat $dir/degs_group.$g.scp | shuffle | round_robin $outputs
    done
  fi
fi

if [ $stage -le 6 ]; then
  echo "$0: getting train-subset scp"
  # get degs_train_subset.scp by taking the top and tail of the degs files [quicker
  # than cat'ing all the files, random shuffling and head]

  nl=$[$num_egs_subset/$num_archives + 1]

  # use utils/shuffle_list.pl because it provides a complete shuffle (ok since
  # the amount of data is small).  note: shuf is not available on mac by
  # default.
  for n in $(seq $num_archives); do
    head -n$nl $dir/degs.$n.scp;  tail -n$nl $dir/degs.$n.scp
  done  | utils/shuffle_list.pl | head -n$num_utts_subset >$dir/train_diagnostic.scp
  [ -s $dir/train_diagnostic.scp ] || { echo "$0: error getting train_diagnostic.scp"; exit 1; }
fi

if [ $stage -le 7 ]; then
  echo "$0: creating final archives"
  $cmd --max-jobs-run "$max_copy_jobs" \
     JOB=1:$num_archives $dir/log/copy_archives.JOB.log \
     nnet3-discriminative-copy-egs scp:$dir/degs.JOB.scp ark:$dir/degs.JOB.ark || exit 1

  run.pl $dir/log/copy_train_subset.log \
      nnet3-discriminative-copy-egs scp:$dir/train_diagnostic.scp \
         ark:$dir/train_diagnostic.degs  || exit 1

  run.pl $dir/log/copy_valid_subset.log \
      nnet3-discriminative-copy-egs scp:$dir/valid_diagnostic.scp \
         ark:$dir/valid_diagnostic.degs  || exit 1
fi

if [ $stage -le 10 ] && $cleanup; then
  echo "$0: cleaning up temporary files."
  for j in $(seq $nj); do
    for f in $dir/degs_orig.$j.{ark,scp} $dir/degs_orig_filtered.$j.scp; do
      [ -L $f ] && rm $(readlink -f $f); rm $f
    done
  done
  rm $dir/degs_group.*.scp $dir/valid_diagnostic.scp $dir/train_diagnostic.scp 2>/dev/null
  rm $dir/ali.ark $dir/ali.scp 2>/dev/null
  for n in $(seq $num_archives); do
    for f in $dir/degs.$n.scp; do
      [ -L $f ] && rm $(readlink -f $f); rm $f
    done
  done
fi


exit 0


echo "$0: Finished decoding and preparing training examples"
