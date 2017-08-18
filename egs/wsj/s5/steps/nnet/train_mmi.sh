#!/bin/bash
# Copyright 2013-2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0.

# Sequence-discriminative MMI/BMMI training of DNN.
# 4 iterations (by default) of Stochastic Gradient Descent with per-utterance updates.
# Boosting of paths with more errors (BMMI) gets activated by '--boost <float>' option.

# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.


# Begin configuration section.
cmd=run.pl
num_iters=4
boost=0.0 #ie. disable boosting
acwt=0.1
lmwt=1.0
learn_rate=0.00001
halving_factor=1.0 #ie. disable halving
drop_frames=true
verbose=0 # 0 No GPU time-stats, 1 with GPU time-stats (slower),
ivector=

seed=777    # seed value used for training data shuffling
skip_cuda_check=false
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# -ne 6 ]; then
  echo "Usage: $0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
  echo " e.g.: $0 data/train_all data/lang exp/tri3b_dnn exp/tri3b_dnn_ali exp/tri3b_dnn_denlats exp/tri3b_dnn_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --num-iters <N>                                  # number of iterations to run"
  echo "  --acwt <float>                                   # acoustic score scaling"
  echo "  --lmwt <float>                                   # linguistic score scaling"
  echo "  --learn-rate <float>                             # learning rate for NN training"
  echo "  --drop-frames <bool>                             # drop frames num/den completely disagree"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"

  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4
denlatdir=$5
dir=$6

for f in $data/feats.scp $denlatdir/lat.scp \
         $alidir/{tree,final.mdl,ali.1.gz} \
         $srcdir/{final.nnet,final.feature_transform}; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# check if CUDA compiled in,
if ! $skip_cuda_check; then cuda-compiled || { echo "Error, CUDA not compiled-in!"; exit 1; } fi

mkdir -p $dir/log

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt
cp $lang/phones.txt $dir

cp $alidir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl`


#Get the files we will need
nnet=$srcdir/$(readlink $srcdir/final.nnet || echo final.nnet);
[ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;
cp $nnet $dir/0.nnet; nnet=$dir/0.nnet

class_frame_counts=$srcdir/ali_train_pdf.counts
[ -z "$class_frame_counts" ] && echo "Error class_frame_counts '$class_frame_counts' does not exist!" && exit 1;
cp $srcdir/ali_train_pdf.counts $dir

feature_transform=$srcdir/final.feature_transform
if [ ! -f $feature_transform ]; then
  echo "Missing feature_transform '$feature_transform'"
  exit 1
fi
cp $feature_transform $dir/final.feature_transform

model=$dir/final.mdl
[ -z "$model" ] && echo "Error transition model '$model' does not exist!" && exit 1;



# Shuffle the feature list to make the GD stochastic!
# By shuffling features, we have to use lattices with random access (indexed by .scp file).
cat $data/feats.scp | utils/shuffle_list.pl --srand $seed >$dir/train.scp

###
### PREPARE FEATURE EXTRACTION PIPELINE
###
# import config,
cmvn_opts=
delta_opts=
D=$srcdir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,o:copy-feats scp:$dir/train.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  [ -z $ivector ] && echo "Missing --ivector, they were used in training!" && exit 1
  # Get the tool,
  ivector_append_tool=append-vector-to-feats # default,
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  # Check dims,
  dim_raw=$(feat-to-dim "$feats" -)
  dim_raw_and_ivec=$(feat-to-dim "$feats $ivector_append_tool ark:- '$ivector' ark:- |" -)
  dim_ivec=$((dim_raw_and_ivec - dim_raw))
  [ $dim_ivec != "$(cat $D/ivector_dim)" ] && \
    echo "Error, i-vector dim. mismatch (expected $(cat $D/ivector_dim), got $dim_ivec in '$ivector')" && \
    exit 1
  # Append to feats,
  feats="$feats $ivector_append_tool ark:- '$ivector' ark:- |"
fi

### Record the setup,
[ ! -z "$cmvn_opts" ] && echo $cmvn_opts >$dir/cmvn_opts
[ ! -z "$delta_opts" ] && echo $delta_opts >$dir/delta_opts
[ -e $D/pytel_transform.py ] && cp $D/pytel_transform.py $dir/pytel_transform.py
[ -e $D/ivector_dim ] && cp $D/ivector_dim $dir/ivector_dim
[ -e $D/ivector_append_tool ] && cp $D/ivector_append_tool $dir/ivector_append_tool
###

###
### Prepare the alignments
###
# Assuming all alignments will fit into memory
ali="ark:gunzip -c $alidir/ali.*.gz |"


###
### Prepare the lattices
###
# The lattices are indexed by SCP (they are not gziped because of the random access in SGD)
lats="scp:$denlatdir/lat.scp"

# Optionally apply boosting
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  # make lattice scp with same order as the shuffled feature scp,
  awk '{ if(r==0) { utt_id=$1; latH[$1]=$0; } # lat.scp
         if(r==1) { if(latH[$1] != "") { print latH[$1]; } } # train.scp
  }' r=0 $denlatdir/lat.scp r=1 $dir/train.scp > $dir/lat.scp
  # get the list of alignments,
  ali-to-phones $alidir/final.mdl "$ali" ark,t:- | awk '{print $1;}' > $dir/ali.lst
  # remove from features sentences which have no lattice or no alignment,
  # (so that the mmi training tool does not blow-up due to lattice caching),
  mv $dir/train.scp $dir/train.scp_unfilt
  awk '{ if(r==0) { latH[$1]="1"; } # lat.scp
         if(r==1) { aliH[$1]="1"; } # ali.lst
         if(r==2) { if((latH[$1] != "") && (aliH[$1] != "")) { print $0; } } # train.scp_
  }' r=0 $dir/lat.scp r=1 $dir/ali.lst r=2 $dir/train.scp_unfilt > $dir/train.scp
  # create the lat pipeline,
  lats="ark,o:lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl scp:$dir/lat.scp '$ali' ark:- |"
fi
###
###
###

# Run several iterations of the MMI/BMMI training
cur_mdl=$nnet
x=1
while [ $x -le $num_iters ]; do
  echo "Pass $x (learnrate $learn_rate)"
  if [ -f $dir/$x.nnet ]; then
    echo "Skipped, file $dir/$x.nnet exists"
  else
    $cmd $dir/log/mmi.$x.log \
     nnet-train-mmi-sequential \
       --feature-transform=$feature_transform \
       --class-frame-counts=$class_frame_counts \
       --acoustic-scale=$acwt \
       --lm-scale=$lmwt \
       --learn-rate=$learn_rate \
       --drop-frames=$drop_frames \
       --verbose=$verbose \
       $cur_mdl $alidir/final.mdl "$feats" "$lats" "$ali" $dir/$x.nnet
  fi
  cur_mdl=$dir/$x.nnet

  #report the progress
  grep -B 2 MMI-objective $dir/log/mmi.$x.log | sed -e 's|^[^)]*)[^)]*)||'

  x=$((x+1))
  learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")

done

(cd $dir; [ -e final.nnet ] && unlink final.nnet; ln -s $((x-1)).nnet final.nnet)

echo "MMI/BMMI training finished"

if [ -e $dir/prior_counts ]; then
  echo "Priors are already re-estimated, skipping... ($dir/prior_counts)"
else
  echo "Re-estimating priors by forwarding 10k utterances from training set."
  . cmd.sh
  nj=$(cat $alidir/num_jobs)
  steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $nj \
    ${ivector:+ --ivector "$ivector"} $data $dir
fi

echo "$0: Done. '$dir'"
exit 0
