#!/usr/bin/env bash

# this script has common stages shared across AMI chain recipes
set -e

# configs for 'chain'
stage=0
mic=ihm
use_ihm_ali=false
# chain options
frames_per_eg=150
max_wer=

# output directory names
dir=
treedir=
lang=
min_seg_len=
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ -z $treedir ] && echo "Set --treedir, this specifies the directory to store new tree " && exit 1;
[ -z $lang ] && echo "Set --lang, this specifies the new lang directory which will have the new topology" && exit 1;
[ -z $dir ] && echo "Set --dir, this specifies the experiment directory to store files relevant to the experiment " && exit 1;

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 10" if you have already
# run those things.

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --use-ihm-ali $use_ihm_ali \
                                  --use-sat-alignments true || exit 1;


# Set the variables. These are based on variables set by run_ivector_common.sh
gmm=tri4a
if [ $use_ihm_ali == "true" ]; then
  gmm_dir=exp/ihm/$gmm
  mic=${mic}_cleanali
  ali_dir=${gmm_dir}_${mic}_train_parallel_sp_ali
  lat_dir=${gmm_dir}_${mic}_train_parallel_sp_lats
else
  gmm_dir=exp/$mic/$gmm
  ali_dir=${gmm_dir}_${mic}_train_sp_ali
  lat_dir=${gmm_dir}_${mic}_train_sp_lats
fi

train_set=train_sp
latgen_train_set=train_sp
if [ $use_ihm_ali == "true" ]; then
  latgen_train_set=train_parallel_sp
fi

###################################

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --cmd "$train_cmd" 4200 data/$mic/$latgen_train_set $lang $ali_dir $treedir

fi

# combining the segments in training data to have a minimum length of frames_per_eg + tolerance
# this is critical stage in AMI (gives 1% absolute improvement)
if [ -z $min_seg_len ]; then
  min_seg_len=$(python -c "print ($frames_per_eg+5)/100.0")
fi

if [ $stage -le 12 ]; then
  rm -rf data/$mic/${train_set}_min${min_seg_len}_hires
  utils/data/combine_short_segments.sh \
      data/$mic/${train_set}_hires $min_seg_len data/$mic/${train_set}_min${min_seg_len}_hires

  #extract ivectors for the new data
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/$mic/${train_set}_min${min_seg_len}_hires data/$mic/${train_set}_min${min_seg_len}_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/$mic/${train_set}_min${min_seg_len}_hires_max2 \
    exp/$mic/nnet3/extractor \
    exp/$mic/nnet3/ivectors_${train_set}_min${min_seg_len} || exit 1;

 # combine the non-hires features for alignments/lattices
 rm -rf data/$mic/${latgen_train_set}_min${min_seg_len}
 utils/data/combine_short_segments.sh \
     data/$mic/${latgen_train_set} $min_seg_len data/$mic/${latgen_train_set}_min${min_seg_len}
fi

train_set=${train_set}_min${min_seg_len}
latgen_train_set=${latgen_train_set}_min${min_seg_len}
ivector_dir=exp/$mic/nnet3/ivectors_${train_set}
ali_dir=${ali_dir}_min${min_seg_len}
lat_dir=${lat_dir}_min${min_seg_len}
if [ $stage -le 13 ]; then
  # realigning data as the segments would have changed
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set data/lang $gmm_dir $ali_dir || exit 1;
fi

if [ $stage -le 14 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

mkdir -p $dir
train_data_dir=data/$mic/${train_set}_hires
if [ ! -z $max_wer ]; then
  if [ $stage -le 15 ]; then
    bad_utts_dir=${gmm_dir}_${mic}_${train_set}_bad_utts # added mic in name as this can be ihm directory where parallel mdm and sdm utts are written
    if [ ! -f $bad_utts_dir/all_info.sorted.txt ]; then
      # This stage takes a lot of time ~7hrs, so run only if file is not available already
      steps/cleanup/find_bad_utts.sh --cmd "$decode_cmd" --nj 405 data/$mic/$latgen_train_set data/lang $ali_dir $bad_utts_dir
    fi
    python local/sort_bad_utts.py --bad-utt-info-file $bad_utts_dir/all_info.sorted.txt --max-wer $max_wer --output-file $dir/wer_sorted_utts_${max_wer}wer
    utils/copy_data_dir.sh --validate-opts "--no-wav"  data/$mic/${train_set}_hires data/$mic/${train_set}_${max_wer}wer_hires
    utils/filter_scp.pl $dir/wer_sorted_utts_${max_wer}wer data/$mic/${train_set}_hires/feats.scp  > data/$mic/${train_set}_${max_wer}wer_hires/feats.scp
    utils/fix_data_dir.sh data/$mic/${train_set}_${max_wer}wer_hires
  fi
  train_data_dir=data/$mic/${train_set}_${max_wer}wer_hires
  # we don't realign again as the segment ids don't change
fi

cat > $dir/vars <<EOF
train_data_dir=$train_data_dir
train_ivector_dir=$ivector_dir
lat_dir=$lat_dir
EOF

exit 0;
