#!/bin/bash

. ./cmd.sh
. ./path.sh

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdmN [single distant microphone- the current script allows you to select
#        any of 4 PZM microphones, D1...D4 in a diagram, best results are with D2]
#  mdm4 [multiple distant microphones-- currently we only support averaging over
#       the 2,3 or 4 single microphones].
# ... by calling this script as, for example,
# ./run.sh --mic ihm  (will build ihm systems from individual headset mics)
# ./run.sh --mic sdm4 (will build sdm systems from D2 mic - look ../README.txt if confused)
# ./run.sh --mic mdm4 (will build mdm systems from D1...D4 mics)
mic=ihm

# Train systems,
nj=30 # number of parallel jobs,
stage=0
. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

set -euo pipefail

# Path where ICSI gets downloaded (or where locally available):
# Note: provide the path to a subdirectory with meeting folders (i.e. B* ones)
ICSI_DIR=/disks/data1/corpora/meeting_speech/speech # Default

[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=$(cat data/local/lm/final_lm)
LM=$final_lm.pr1-7

# This recipe assumes (so far) you obtained the corpus already (can do so from LDC or http://groups.inf.ed.ac.uk/ami/icsi/)

if [[ "$base_mic" =~ "mdm" ]]; then
  echo "Running multi distant channel recipe with beamforming...."
  PROCESSED_ICSI_DIR=$ICSI_DIR/../beamformed
  if [ $stage -le 1 ]; then
    # for MDM data, do beamforming
    ! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; extras/install_beamformit.sh; cd -;'" && exit 1
    local/icsi_beamform.sh --cmd "$train_cmd" --nj 20 $mic $ICSI_DIR $PROCESSED_ICSI_DIR
  fi
else
  PROCESSED_ICSI_DIR=$ICSI_DIR
fi

# Prepare original data directories data/ihm/train_orig, etc.
if [ $stage -le 2 ]; then
  local/icsi_${base_mic}_data_prep.sh $PROCESSED_ICSI_DIR $mic
  local/icsi_${base_mic}_scoring_data_prep.sh $PROCESSED_ICSI_DIR $mic dev
  local/icsi_${base_mic}_scoring_data_prep.sh $PROCESSED_ICSI_DIR $mic eval
fi

if [ $stage -le 3 ]; then
  for dset in train dev eval; do
    # this splits up the speakers (which for sdm and mdm just correspond
    # to recordings) into 30-second chunks.  It's like a very brain-dead form
    # of diarization; we can later replace it with 'real' diarization.
    seconds_per_spk_max=30
    [ "$mic" == "ihm" ] && seconds_per_spk_max=120  # speaker info for ihm is real,
                                                    # so organize into much bigger chunks.

    # Note: the 30 on the next line should have been $seconds_per_spk_max
    # (thanks: Pavel Denisov.  This is a bug but before fixing it we'd have to
    # test the WER impact.  I suspect it will be quite small and maybe hard to
    # measure consistently.
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 \
      data/$mic/${dset}_orig data/$mic/$dset
  done
fi

# Feature extraction,
if [ $stage -le 4 ]; then
  for dset in train dev eval; do
    steps/make_mfcc.sh --nj 15 --cmd "$train_cmd" data/$mic/$dset
    steps/compute_cmvn_stats.sh data/$mic/$dset
    utils/fix_data_dir.sh data/$mic/$dset
  done
fi

# monophone training
if [ $stage -le 5 ]; then
  # Full set 77h, reduced set 10.8h,
  utils/subset_data_dir.sh data/$mic/train 15000 data/$mic/train_15k

  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train_15k data/lang exp/$mic/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/mono exp/$mic/mono_ali
fi

# context-dep. training with delta features.
if [ $stage -le 6 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 80000 data/$mic/train data/lang exp/$mic/mono_ali exp/$mic/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri1 exp/$mic/tri1_ali
fi

if [ $stage -le 7 ]; then
  # LDA_MLLT
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/$mic/train data/lang exp/$mic/tri1_ali exp/$mic/tri2
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri2 exp/$mic/tri2_ali
  # Decode
  graph_dir=exp/$mic/tri2/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri2 $graph_dir
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri2/decode_dev_${LM}
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri2/decode_eval_${LM}
fi


if [ $stage -le 8 ]; then
  # LDA+MLLT+SAT
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 80000 data/$mic/train data/lang exp/$mic/tri2_ali exp/$mic/tri3
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri3 exp/$mic/tri3_ali
fi

if [ $stage -le 9 ]; then
  # Decode the fMLLR system.
  graph_dir=exp/$mic/tri3/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri3 $graph_dir
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri3/decode_dev_${LM}
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri3/decode_eval_${LM}
fi

if [ $stage -le 10 ]; then
  # The following script cleans the data and produces cleaned data
  # in data/$mic/train_cleaned, and a corresponding system
  # in exp/$mic/tri3_cleaned.  It also decodes.
  #
  # Note: local/run_cleanup_segmentation.sh defaults to using 50 jobs,
  # you can reduce it using the --nj option if you want.
  #local/run_cleanup_segmentation.sh --mic $mic
  echo "For ICSI we do not clean segmentations, as those are manual by default, so should be OK."
  #but perhaps running such experiment would make sense, for now I want to keep this recipe as
  #close to the baseline one as possibe
fi

if [ $stage -le 11 ]; then
  ali_opt=
  [ "$mic" != "ihm" ] && ali_opt="--use-ihm-ali true"
  local/chain/run_tdnn.sh $ali_opt --mic $mic
fi

#if [ $stage -le 12 ]; then
#  the following shows how you would run the nnet3 system; we comment it out
#  because it's not as good as the chain system.
#  ali_opt=
#  [ "$mic" != "ihm" ] && ali_opt="--use-ihm-ali false"
#  local/nnet3/run_tdnn.sh $ali_opt --mic $mic 
#fi

exit 0
