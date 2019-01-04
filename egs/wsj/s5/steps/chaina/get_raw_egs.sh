#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This script dumps 'raw' egs for 'chaina' training.  What 'raw' means in this
# context is that they need to be further processed to merge egs of the same
# speaker, etc.  So they won't be directly consumed by training, but by
# by the script process_egs.sh.



# Begin configuration section.
cmd=run.pl
frames_per_chunk=150  # Number of frames (at feature frame rate) per example.  You
                      # are allowed to make this a comma-separated list,
                      # e.g. 150,110,100, meaning that a range of eg widths are
                      # allowed (but this may not be as helpful when using our
                      # adaptation framework, since it will tend to split up
                      # utterances into separate minibatches.

frame_subsampling_factor=3 # frames-per-second of features we train on divided
                           # by frames-per-second at output of chain model
alignment_subsampling_factor=3 # frames-per-second of input alignments divided
                               # by frames-per-second at output of chain model
constrained=true  # 'constrained=true' is the traditional setup; 'constrained=false'
                  # gives you the 'unconstrained' egs creation in which the time
                  # boundaries are not enforced inside chunks.
left_context=0    # amount of left-context per eg (i.e. extra frames of input
                  # features not present in the output supervision).  Would
                  # normally depend on the model context, plus desired 'extra'
                  # context (e.g. for LSTM).
right_context=0   # amount of right-context per eg.

left_context_initial=-1   # if >=0, right-context for last chunk of an utterance.
right_context_final=-1     # if >=0, right-context for last chunk of an utterance.

compress=true   # set this to false to disable compression (e.g. if you want to
                # see whether results are affected).  Note: if the features on
                # disk were originally compressed, nnet3-chain-get-egs will dump
                # compressed features regardless (since there is no further loss
                # in that case).

lang=default   # the language name.  will usually be 'default' in single-language
               # setups.  Requires because it's part of the name of some of
               # the input files.

right_tolerance=  # chain right tolerance == max label delay.  Only relevant if
                  # constrained=true.  At frame rate of alignments.  Code
                  # default is 5.
left_tolerance=   # chain left tolerance (versus alignments from lattices).
                  # Only relevant if constrained=true.  At frame rate of
                  # alignments.  Code default is 5.

stage=0
max_jobs_run=40         # This should be set to the maximum number of
                        # nnet3-chain-get-egs jobs you are comfortable to run in
                        # parallel; you can increase it if your disk speed is
                        # greater and you have more machines.


srand=0         # rand seed for nnet3-chain-get-egs, nnet3-chain-copy-egs and nnet3-chain-shuffle-egs

lattice_lm_scale=     # If supplied, the graph/lm weight of the lattices will be
                      # used (with this scale) in generating supervisions
                      # This is 0 by default for conventional supervised training,
                      # but may be close to 1 for the unsupervised part of the data
                      # in semi-supervised training. The optimum is usually
                      # 0.5 for unsupervised data.
lattice_prune_beam=        # If supplied, the lattices will be pruned to this beam,
                           # before being used to get supervisions.

acwt=0.1   # For pruning.  Should be, for instance, 1.0 for chain lattices.
deriv_weights_scp=


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <chain-dir> <lattice-dir> <raw-egs-dir>"
  echo " e.g.: $0 data/train exp/chaina/tdnn1a_sp exp/tri3_lats exp/chaina/tdnn1a_sp/raw_egs"
  echo ""
  echo "From <chain-dir>, 0/<lang>.mdl (for the transition-model), <lang>.tree (the tree), "
  echo "   den_fsts/<lang>.den.fst, and den_fsts/<lang>.normalization.fst (the normalization "
  echo "   FST, derived from the denominator FST echo are read (where <lang> is specified"
  echo "   by the --lang option (its default values is 'default')"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options (alternative to this"
  echo "                                                   # command line)"
  echo "  --max-jobs-run <max-jobs-run>                    # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --frame-subsampling-factor <factor;3>            # factor by which num-frames at nnet output is reduced "
  echo "  --lang       <language-name;'default'>           # Name of the language, determines names of some inputs."
  echo "  --frames-per-chunk <frames;150>                  # number of supervised frames per chunk on disk"
  echo "                                                   # ... may be a comma separated list, but we advise a single"
  echo "                                                   #  number in most cases, due to interaction with the need "
  echo "                                                   # to group egs from the same speaker into groups."
  echo "  --left-context <int;0>                           # Number of frames on left side to append for feature input"
  echo "  --right-context <int;0>                          # Number of frames on right side to append for feature input"
  echo "  --left-context-initial <int;-1>                  # Left-context for first chunk of an utterance"
  echo "  --right-context-final <int;-1>                   # Right-context for last chunk of an utterance"
  echo "  --lattice-lm-scale <float>                       # If supplied, the graph/lm weight of the lattices will be "
  echo "                                                   # used (with this scale) in generating supervisions"
  echo "  --lattice-prune-beam <float>                     # If supplied, the lattices will be pruned to this beam, "
  echo "                                                   # before being used to get supervisions."
  echo "  --acwt <float;0.1>                               # Acoustic scale -- should be acoustic scale at which the "
  echo "                                                   # supervision lattices are to be interpreted.  Affects pruning"
  echo "  --deriv-weights-scp <str>                        # If supplied, adds per-frame weights to the supervision."
  echo "                                                   # (e.g., might be relevant for unsupervised training)."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi

data=$1
chaindir=$2
latdir=$3
dir=$4

tree=$chaindir/${lang}.tree
trans_mdl=$chaindir/0/${lang}.mdl  # contains the transition model and a nnet.
normalization_fst=$chaindir/0/${lang}.normalization.fst
den_fst=$chaindir/0/${lang}.den.fst

for f in $data/feats.scp $latdir/lat.1.gz $latdir/final.mdl \
         $tree $trans_mdl $normalization_fst $den_fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=$(cat $latdir/num_jobs) || exit 1
if [ -f $latdir/per_utt ]; then
  sdata=$data/split${nj}utt
  utils/split_data.sh --per-utt $data $nj
else
  sdata=$data/split$nj
  utils/split_data.sh $data $nj
fi

mkdir -p $dir/log  $dir/misc

cp $tree $dir/misc/
copy-transition-model $trans_mdl $dir/misc/${lang}.trans_mdl
cp $normalization_fst $den_fst $dir/misc/
cp data/utt2spk $dir/misc/
if [ -f $data/utt2uniq ]; then
  cp $data/utt2uniq $dir/misc/
elif [ -f $dir/misc/utt2uniq ]; then
  rm $dir/misc/utt2uniq
fi

if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $nj); do echo $dir/cegs.$x.ark; done)
fi


lats_rspecifier="ark:gunzip -c $latdir/lat.JOB.gz |"
if [ ! -z $lattice_prune_beam ]; then
  if [ "$lattice_prune_beam" == "0" ] || [ "$lattice_prune_beam" == "0.0" ]; then
    lats_rspecifier="$lats_rspecifier lattice-1best --acoustic-scale=$acwt ark:- ark:- |"
  else
    lats_rspecifier="$lats_rspecifier lattice-prune --acoustic-scale=$acwt --beam=$lattice_prune_beam ark:- ark:- |"
  fi
fi

egs_opts="--long-key=true --left-context=$left_context --right-context=$right_context --num-frames=$frames_per_chunk --frame-subsampling-factor=$frame_subsampling_factor --compress=$compress"
[ $left_context_initial -ge 0 ] && egs_opts="$egs_opts --left-context-initial=$left_context_initial"
[ $right_context_final -ge 0 ] && egs_opts="$egs_opts --right-context-final=$right_context_final"

[ ! -z "$deriv_weights_scp" ] && egs_opts="$egs_opts --deriv-weights-rspecifier=scp:$deriv_weights_scp"


chain_supervision_all_opts="--lattice-input=true --frame-subsampling-factor=$alignment_subsampling_factor"
[ ! -z $right_tolerance ] && \
  chain_supervision_all_opts="$chain_supervision_all_opts --right-tolerance=$right_tolerance"

[ ! -z $left_tolerance ] && \
  chain_supervision_all_opts="$chain_supervision_all_opts --left-tolerance=$left_tolerance"

if ! $constrained; then
  # e2e supervision
  chain_supervision_all_opts="$chain_supervision_all_opts --convert-to-pdfs=false"
  egs_opts="$egs_opts --transition-model=$chaindir/0.trans_mdl"
fi

if [ ! -z "$lattice_lm_scale" ]; then
  chain_supervision_all_opts="$chain_supervision_all_opts --lm-scale=$lattice_lm_scale"

  normalization_fst_scale=$(perl -e "
  if ($lattice_lm_scale >= 1.0 || $lattice_lm_scale < 0) {
    print STDERR \"Invalid --lattice-lm-scale $lattice_lm_scale\"; exit(1);
  }
  print (1.0 - $lattice_lm_scale);") || exit 1
  egs_opts="$egs_opts --normalization-fst-scale=$normalization_fst_scale"
fi

if [ $stage -le 0 ]; then
  $cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/get_egs.JOB.log \
       lattice-align-phones --replace-output-symbols=true $latdir/final.mdl \
       "$lats_rspecifier" ark:- \| \
       chain-get-supervision $chain_supervision_all_opts \
       $dir/misc/tree $dir/misc/${lang}.trans_mdl ark:- ark:- \| \
       nnet3-chain-get-egs $ivector_opts --srand=\$[JOB+$srand] $egs_opts \
       "$normalization_fst" $sdata/JOB/feats.scp ark,s,cs:- \
       ark,scp:$dir/cegs.JOB.ark,$dir/cegs.JOB.scp || exit 1;
fi


if [ $stage -le 1 ]; then
  frames_and_chunks=$(for n in $(seq nj); do cat $dir/log/get_egs.$n.log; done | \
                      perl -e '$nf=0;$nc=0; while(<STDIN>) { if(m/with total length (\d+) frames.+ into (\d+) chunks/) { $nf += $1; $nc += $2; } print "$nf $nc";')
  num_frames=$(echo $frames_and_chunks || awk '{print $1}')
  num_chunks=$(echo $frames_and_chunks || awk '{print $2}')
  frames_per_chunk_avg=$[$num_frames/$num_chunks]
  feat_dim=$(feat-to-dim scp:$sdata/JOB/feats.scp -)
  num_leaves=$(tree-info $chaindir/tree | awk '/^num-pdfs/ {print $2}')
  if [ $left_context_initial -lt 0 ]; then
    left_context_initial=$left_context
  fi
  if [ $right_context_final -lt 0 ]; then
    right_context_initial=$right_context
  fi

  cat >$dir/info.txt <<EOF
dir_type raw_chaina_egs
num_input_frames $num_frames
num_chunks $num_chunks
lang $lang
feat_dim $feat_dim
num_leaves $num_leaves
frames_per_chunk $frames_per_chunk
frames_per_chunk_avg $frames_per_chunk_avg
left_context $left_context
left_context_initial $left_context_initial
right_context $right_context
right_context_final $right_context_final
EOF

  if ! cat $dir/info.txt | awk '{if (NF == 1) exit(1);}'; then
    echo "$0: we failed to obtain at least one of the fields in $dir/info.txt"
    exit 1
  fi
fi


if [ $stage -le 2 ]; then
  for n in $(seq $nj); do cat $dir/cegs.$n.scp; done > $dir/all.scp
fi

echo "$0: Finished preparing raw egs"
