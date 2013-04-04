#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# MMI training (or optionally boosted MMI, if you give the --boost option).
# 4 iterations (by default) of Extended Baum-Welch update.
#
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
do_smbr=true
use_silphones=false #setting this to something will enable giving siphones to nnet-mpe
verbose=1
use_gpu_id=

seed=777    # seed value used for training data shuffling
#stage=0
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/$0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
  echo " e.g.: steps/$0 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
#  echo "  --stage <stage>                                  # stage to do partial re-run from."
  
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4
denlatdir=$5
dir=$6
mkdir -p $dir/log

for f in $data/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $denlatdir/lat.scp $srcdir/{final.nnet,final.feature_transform}; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log

cp $alidir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl` || exit 1;



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

#enable/disable silphones from MPE training
mpe_silphones_arg= #empty
[ "$use_silphones" == "true" ] && mpe_silphones_arg="--silence-phones=$silphonelist"


# Shuffle the feature list to make the GD stochastic!
# In doing so, we have to make sure the lattices are either indexed by .scp file
# or are stored in the same order as features, which is harder to do...
# The alignments can fit-in the memory, so no special treatment is necessary for them.
cat $data/feats.scp | utils/shuffle_list.pl --srand $seed > $dir/train.scp


###
### Prepare feature pipeline
###
# Create the feature stream:
feats="ark,s,cs:copy-feats scp:$dir/train.scp ark:- |"
# Optionally add cmvn
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $data/cmvn.scp ] && echo "$0: cannot find cmvn stats $data/cmvn.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  cp $srcdir/norm_vars $dir
fi
# Optionally add deltas
if [ -f $srcdir/delta_order ]; then
  delta_order=$(cat $srcdir/delta_order)
  feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
  cp $srcdir/delta_order $dir
fi
###
###
###


###
### Prepare the alignments
###
ali="ark:gunzip -c $alidir/ali.*.gz |"


###
### Prepare the lattices
###
# The lattices are indexed by SCP (they are no gziped because of the random access in SGD)
lats="scp:$denlatdir/lat.scp"

# Optionally apply boosting
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  #make lattice scp with same order as the shuffled feature scp
  awk '{ if(r==0) { latH[$1]=$2; }
         if(r==1) { if(latH[$1] != "") { print $1" "latH[$1] } }
  }' $denlatdir/lat.scp r=1 $dir/train.scp > $dir/lat.scp
  #get the list of alignments
  ali-to-phones $alidir/final.mdl "$ali" ark,t:- | awk '{print $1;}' > $dir/ali.lst
  #remove feature files which have no lattice or no alignment,
  #(so that the mmi training tool does not blow-up due to lattice caching)
  mv $dir/train.scp $dir/train.scp_unfilt
  awk '{ if(r==0) { latH[$1]="1"; }
         if(r==1) { aliH[$1]="1"; }
         if(r==2) { if((latH[$1] != "") && (aliH[$1] != "")) { print $0; } }
  }' $dir/lat.scp r=1 $dir/ali.lst r=2 $dir/train.scp_unfilt > $dir/train.scp
  #create the lat pipeline
  lats="ark,o:lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl scp:$dir/lat.scp '$ali' ark:- |"
fi
###
###
###

# Run several iterations of the MMI training
cur_mdl=$nnet
x=1
while [ $x -le $num_iters ]; do
  echo "Pass $x (learnrate $learn_rate)"
  if [ -f $dir/$x.nnet ]; then
    echo "Skipped, file $dir/$x.nnet exists"
  else
    #train
    $cmd $dir/log/mpe.$x.log \
     nnet-train-mpe-sequential \
       --feature-transform=$feature_transform \
       --class-frame-counts=$class_frame_counts \
       --acoustic-scale=$acwt \
       --lm-scale=$lmwt \
       --learn-rate=$learn_rate \
       --do-smbr=$do_smbr \
       --verbose=$verbose \
       $mpe_silphones_arg \
       ${use_gpu_id:+ --use-gpu-id=$use_gpu_id} \
       $cur_mdl $alidir/final.mdl "$feats" "$lats" "$ali" $dir/$x.nnet || exit 1
  fi
  cur_mdl=$dir/$x.nnet

  #report the progress
  grep -B 2 "Overall average frame-accuracy" $dir/log/mpe.$x.log | sed -e 's|.*)||'

  x=$((x+1))
  learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
  
done

(cd $dir; [ -e final.nnet ] && unlink final.nnet; ln -s $((x-1)).nnet final.nnet)

echo "MPE training finished"



exit 0




