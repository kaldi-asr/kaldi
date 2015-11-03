#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# Begin configuration.
config=            # config, which is also sent to all other scripts
stage=0

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

# strm options
strm_indices=      # strm indices

# LABELS
labels=            # use these labels to train (override deafault pdf alignments, has to be in 'Posterior' format, see ali-to-post) 
num_tgt=           # force to use number of outputs in the MLP (default is autodetect)

# TRAINING SCHEDULER
learn_rate=0.008   # initial learning rate
train_tool=theano-nnet/nnet1_v2/train_1iter_drop_strms.py # optionally change the training tool
cv_tool=theano-nnet/nnet1_v2/nnet_cross_validate.py # optionally change the training tool

train_cv_tool_opts=     # options, passed to the training and cv tool (shared opts)
train_tool_opts=        # options, passed to the training tool
cv_tool_opts=           # options, passed to the cv tool

frame_weights=     # per-frame weights for gradient weighting

# learn rate scheduling,
max_iters=20
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection,
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5

#which queue?
cv_cmd="${cuda_cmd}"

. utils/parse_options.sh || exit 1;
# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 6 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali_train exp/mono_ali_cv exp/mono_nnet"
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
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

# Using alidir for supervision (default)
if [ -z "$labels" ]; then 
  silphonelist=`cat $lang/phones/silence.csl` || exit 1;
  for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"
printf "\t CV-set    : $data_cv $alidir_cv \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final_nnet.pklz ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final_nnet.pklz ($(readlink $dir/final_nnet.pklz))\n\n" && exit 0

echo "# PREPARING ALIGNMENTS"
if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels"
else
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" ark,t:- > $dir/train.labels.ascii
  labels_tr=$dir/train.labels.ascii
  ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir_cv/ali.*.gz |" ark,t:- > $dir/cv.labels.ascii
  labels_cv=$dir/cv.labels.ascii

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
echo "Preparing train/cv lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

###### PREPARE FEATURE PIPELINE ######
feats_opts=
# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_opts="$feats_opts $delta_opts"
  echo "add-deltas with $delta_opts"
fi
#optionally cmvn_opts
if [ ! -z "$cmvn_opts" ]; then
  feats_opts="$feats_opts $cmvn_opts"
  echo "cmnv-opts with $cmvn_opts"
fi
#add splice and splice-step
feats_opts="$feats_opts --splice=$splice --splice-step=$splice_step"
echo $feats_opts >$dir/feats_opts

###### OPTS DIFF TRAIN AND CV ########
# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data/utt2spk ] && echo "Missing $data/utt2spk" && exit 1;
  [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
  [ ! -r $data_cv/utt2spk ] && echo "Missing $data_cv/utt2spk" && exit 1;
  feats_opts_tr="--utt2spk-file=$data/utt2spk --cmvn-scp=$data/cmvn.scp"
  feats_opts_cv="--utt2spk-file=$data_cv/utt2spk --cmvn-scp=$data_cv/cmvn.scp"
else
  echo "apply-cmvn is not used"
fi

###### CREATE FEAT PREPROCESS ######
if [ ! -e $dir/feat_preprocess.pkl ]; then
python theano-nnet/nnet1_v2/create_feat_preprocess.py \
  $feats_opts \
  ${feats_opts_tr:+ ${feats_opts_tr}} \
  $dir/train.scp $dir/feat_preprocess.pkl 2>$dir/log/create_feat_preprocess.log || exit 1
fi

###### NNET PROTOTYPE ######
if [ -z "$nnet_proto" ]; then
  #output-dim
  [ -z $num_tgt ] && num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

  #input-dim
  num_fea=$(python theano-nnet/nnet1_v2/feat_to_dim.py $feats_opts ${feats_opts_tr:+ ${feats_opts_tr}} $dir/train.scp)

  # make network prototype
  nnet_proto=$dir/nnet.proto
  echo "Creating proto file in $nnet_proto"
  utils/nnet/make_nnet_proto.py $proto_opts \
    ${bn_dim:+ --bottleneck-dim=$bn_dim} \
    $num_fea $num_tgt $hid_layers $hid_dim >$nnet_proto || exit 1 

  ###### NNET INITIALIZE ######
  echo "Initializing with $nnet_proto"
  python theano-nnet/nnet1_v2/nnet_initialize.py  \
    $nnet_proto $dir/nnet_initial.pklz 2>$dir/log/nnet_initialize.log || exit 1

fi


echo ""
echo "##############################"
echo "Started neural net training"

num_strms=`echo $strm_indices | awk -F "," '{print NF-1}'`
tot_comb=`echo "2^$num_strms"-1|bc`

nnet_best=$dir/nnet_initial.pklz
# cross-validation
iter=00
cv_done_file=$dir/.done_cv_iter${iter}
if [ ! -e $cv_done_file ]; then

echo "Cross-validating using INITIAL $nnet_best"
echo " total combinatios = $tot_comb"
$cv_cmd JOB=1:$tot_comb $dir/log/iter${iter}_comb.JOB.cv.log \
theano-nnet/nnet1_v2/cross_validate.sh \
  --cv-tool $cv_tool \
  --feat-preprocess $dir/feat_preprocess.pkl \
  --tool-opts "$train_cv_tool_opts $cv_tool_opts --strm-indices=$strm_indices --comb-num=JOB $feats_opts_cv --done-file=$dir/.done_cv_iter${iter}_comb.JOB" \
  $dir/cv.scp $labels_cv $nnet_best || exit 1;

#Combine them to $cv_done_file
python theano-nnet/nnet1_v2/combine_done_files.py \
  $cv_done_file $dir/.done_cv_iter${iter}_comb.* 2>$dir/log/combine_done_files.log || exit 1;
#clean-up
rm $dir/.done_cv_iter${iter}_comb.*
fi

# optionally resume training from the best epoch, using saved learning-rate,
[ -e $dir/.best_nnet ] && nnet_best=$(cat $dir/.best_nnet)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

[ ! -e $dir/.best_cv ] && cp $dir/.done_cv_iter00 $dir/.best_cv

echo $learn_rate >$dir/.learn_rate
echo $nnet_best >$dir/.best_nnet

##############################
# start training
for iter in $(seq -w $max_iters); do

  learn_rate=$(cat $dir/.learn_rate)
  nnet_best=$(cat $dir/.best_nnet)

  echo "##############################"
  echo "ITERATION $iter: "
  nnet_next=$dir/nnet/nnet_iter${iter}
  
  train_done_file=$dir/.done_train_iter${iter}
  if [ ! -e $train_done_file ]; then
    echo "training ITERATION $iter"
    log=$dir/log/iter${iter}.tr.log
    $cuda_cmd $log \
    theano-nnet/nnet1_v2/train_1iter.sh \
      --train-tool $train_tool \
      --feat-preprocess $dir/feat_preprocess.pkl \
      --tool-opts "$train_cv_tool_opts $train_tool_opts --strm-indices=$strm_indices --learn-rate=$learn_rate $feats_opts_tr --done-file=$train_done_file" \
    $dir/train.scp $labels_tr $nnet_best $nnet_next || exit 1;
  else
    echo "Skipping training ITERATION $iter"
  fi

  # cross-validation
  cv_done_file=$dir/.done_cv_iter${iter}
  if [ ! -e $cv_done_file ]; then
    echo "Cross-validating using INITIAL $nnet_best"
    echo " total combinatios = $tot_comb"
    $cv_cmd JOB=1:$tot_comb $dir/log/iter${iter}_comb.JOB.cv.log \
    theano-nnet/nnet1_v2/cross_validate.sh \
      --cv-tool $cv_tool \
      --feat-preprocess $dir/feat_preprocess.pkl \
      --tool-opts "$train_cv_tool_opts $cv_tool_opts --strm-indices=$strm_indices --comb-num=JOB $feats_opts_cv --done-file=$dir/.done_cv_iter${iter}_comb.JOB" \
    $dir/cv.scp $labels_cv $nnet_next || exit 1;

    #Combine them to $cv_done_file
    python theano-nnet/nnet1_v2/combine_done_files.py \
      $cv_done_file $dir/.done_cv_iter${iter}_comb.* 2>$dir/log/combine_done_files.log || exit 1;
    #clean-up
    rm $dir/.done_cv_iter${iter}_comb.*

    #Estimate learn-rate and best-nnet
    #for this iteration
    sleep 60
    python theano-nnet/nnet1_v2/newbob_schedule.py \
      --iter=$iter \
      --learn-rate=$learn_rate \
      --start-halving-impr=$start_halving_impr \
      --end-halving-impr=$end_halving_impr \
      --halving-factor=$halving_factor \
    $dir || exit 1;

  else
    echo "Skipping Cross-validating ITERATION $iter"
  fi

  if [ -e $dir/.finished ]; then
    break
  fi
done

# select the best network,
if [ $nnet_best != $dir/nnet_initial.pklz ]; then 
  nnet_final=${nnet_best}_final_
  ( cd $dir/nnet; ln -s $(basename $nnet_best) $(basename $nnet_final); )
  ( cd $dir; ln -s nnet/$(basename $nnet_final) final_nnet.pklz; )
  echo "Succeeded training the Neural Network : $dir/final_nnet.pklz"
else
  "Error training neural network..."
  exit 1
fi

echo "$0 successfuly finished.. $dir"
sleep 3
exit 0


