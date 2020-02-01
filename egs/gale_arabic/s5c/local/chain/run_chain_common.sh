#!/usr/bin/env bash

# this script has common stages shared across librispeech chain recipes.
# It generates a new topology in a new lang directory, gets the alignments as
# lattices, and builds a tree for the new topology
set -e

stage=11

# input directory names. These options are actually compulsory, and they have
# been named for convenience
gmm_dir=
ali_dir=
lores_train_data_dir=

num_leaves=6000

# output directory names. They are also compulsory.
lang=
lat_dir=
tree_dir=
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ -z $lang ] && echo "Set --lang, this specifies the new lang directory which will have the new topology" && exit 1;
[ -z $lat_dir ] && echo "Set --lat-dir, this specifies the experiment directory to store lattice" && exit 1;
[ -z $tree_dir ] && echo "Set --tree-dir, this specifies the directory to store new tree " && exit 1;

for f in $gmm_dir/final.mdl $ali_dir/ali.1.gz $lores_train_data_dir/feats.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 11 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 12 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat ${ali_dir}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    $lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 13 ]; then
  # Build a tree using our new topology. We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" $num_leaves ${lores_train_data_dir} $lang $ali_dir $tree_dir
fi

exit 0;
