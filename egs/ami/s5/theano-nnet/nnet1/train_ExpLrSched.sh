#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
nnet_init=          # select initialized MLP (override initialization) (TODO)
nnet_proto=
proto_opts=        # non-default options for 'make_nnet_proto.py'
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
pytel_transform=   # use external transform defined in python (BUT specific)
network_type=dnn   # (dnn,cnn1d,cnn2d,lstm) select type of neural network
cnn_proto_opts=     # extra options for 'make_cnn_proto.py'
#
#
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
hid_dim=1024       # select hidden dimension
bn_dim=            # set a value to get a bottleneck network
dbn=               # select DBN to prepend to the MLP initialization
#
init_opts=         # options, passed to the initialization script

# FEATURE PROCESSING
copy_feats=true # resave the train/cv features into /tmp (disabled by default)
 copy_feats_tmproot= # tmproot for copy-feats (optional)
# feature config (applies always)
cmvn_opts=
delta_opts=
# feature_transform:
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
feat_type=plain
# feature config (applies to feat_type traps)
traps_dct_basis=11 # nr. od DCT basis (applies to `traps` feat_type, splice10 )
# feature config (applies to feat_type transf) (ie. LDA+MLLT, no fMLLR)
transf=
splice_after_transf=5
# feature config (applies to feat_type lda)
lda_dim=300        # LDA dimension (applies to `lda` feat_type)

# LABELS
labels=            # use these labels to train (override deafault pdf alignments, has to be in 'Posterior' format, see ali-to-post) 
num_tgt=           # force to use number of outputs in the MLP (default is autodetect)

# TRAINING SCHEDULER
learn_rate=0.008   # initial learning rate
train_tool=theano-nnet/nnet1/train_ExpLrSched.py # optionally change the training tool
train_opts=        # options, passed to the training script
frame_weights=     # per-frame weights for gradient weighting


. utils/parse_options.sh || exit 1;
# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 4 ]; then
   echo "Usage: $0 <data-train> <lang-dir> <ali-train> <exp-dir>"
   echo " e.g.: $0 data/train data/lang exp/mono_ali_train exp/mono_nnet"
   echo ""
   echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate/model selection based on cross-entopy)"
   echo " note.: <ali-train>,<ali-dev> can point to same directory, or 2 separate directories."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --copy-feats <bool>      # copy input features to /tmp (it's faster)"
   echo ""
   exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

# Using alidir for supervision (default)
if [ -z "$labels" ]; then 
  silphonelist=`cat $lang/phones/silence.csl` || exit 1;
  for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final_nnet.pklz ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final_nnet.pklz ($(readlink $dir/final_nnet.pklz))\n\n" && exit 0

  
echo "# PREPARING ALIGNMENTS"
if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
else
  echo "Using PDF targets from dirs '$alidir'"
  ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" ark,t:- > $dir/train.labels.ascii
  labels_tr=$dir/train.labels.ascii

  labels_tr_pdf="ark:ali-to-pdf ${alidir}/final.mdl \"ark:gunzip -c ${alidir}/ali.*.gz |\" ark:- |" # for analyze-counts.
  # get pdf-counts, used later to post-process DNN posteriors
  analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1

  echo "Copying the old transition model, will be needed by decoder"
  copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1

  # copy the tree
  cp $alidir/tree $dir/tree || exit 1
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
# print the list sizes
wc -l $dir/train.scp

# re-save the train features to /tmp, reduces LAN traffic, avoids disk-seeks due to shuffled features
if [ "$copy_feats" == "true" ]; then
  tmpdir=$(mktemp -d $copy_feats_tmproot); mv $dir/train.scp{,_non_local}
  copy-feats scp:$dir/train.scp_non_local ark,scp:$tmpdir/train.ark,$dir/train.scp || exit 1
  trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
fi

feats_opts=
###### PREPARE FEATURE PIPELINE ######
# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data/utt2spk ] && echo "Missing $data/utt2spk" && exit 1;
  feats_opts="$feats_opts $cmvn_opts" 
  feats_opts="$feats_opts --trn-utt2spk-file=$data/utt2spk --trn-cmvn-scp=$data/cmvn.scp"
else
  echo "apply-cmvn is not used"
fi
# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_opts="$feats_opts $delta_opts"
  echo "add-deltas with $delta_opts"
fi
#add splice and splice-step
feats_opts="$feats_opts --splice=$splice --splice-step=$splice_step"
echo $feats_opts >$dir/feats_opts

###### NNET PROTOTYPE ######
if [ -z "$nnet_proto" ]; then
  #output-dim
  [ -z $num_tgt ] && num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

  #input-dim
  num_fea=$(python theano-nnet/nnet1/feat_to_dim.py $feats_opts $dir/train.scp)

  # make network prototype
  nnet_proto=$dir/nnet.proto
  echo "Creating proto file in $nnet_proto"
  utils/nnet/make_nnet_proto.py $proto_opts \
    ${bn_dim:+ --bottleneck-dim=$bn_dim} \
    $num_fea $num_tgt $hid_layers $hid_dim >$nnet_proto || exit 1 

  train_opts="$train_opts --nnet-proto=$nnet_proto"
fi

echo "Training neural network"
python $train_tool \
  $feats_opts ${train_opts:+ ${train_opts}} \
  ${config:+ --config=$config} \
  $dir/train.scp $labels_tr $dir || exit 1;

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0


