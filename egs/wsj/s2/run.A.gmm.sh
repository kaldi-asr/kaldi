#!/bin/bash

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# might want to run this script on a machine that has plenty of memory.

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.

#1)BUT
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
#cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
cuda_cmd="queue.pl -q long.q@pcspeech-gpu"
mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

#2)JHU
#export train_cmd="queue.pl -q all.q@a*.clsp.jhu.edu -S /bin/bash"
#export decode_cmd="queue.pl -q all.q@a*.clsp.jhu.edu -S /bin/bash"

#3)LOCAL
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl


# put the scripts to path
source path.sh


######################################################
### PREPARE THE DATA-SET                           ###
######################################################

# The next command line is an example; you have to give the script
# command-line arguments corresponding to the WSJ disks from LDC.  
# Another example set of command line arguments is
# /ais/gobi2/speech/WSJ/*/??-{?,??}.?
#  These must be absolute,  not relative, pathnames.

local/wsj_data_prep.sh /mnt/matylda2/data/WSJ?/??-{?,??}.? || exit 1;
#local/wsj_data_prep.sh  /export/corpora5/LDC/LDC{93S6,94S13}B/??-{?,??}.? || exit 1;

local/wsj_prepare_dict.sh || exit 1;

local/wsj_format_data.sh || exit 1;



# Now make MFCC features.
# featdir should be some place with a largish disk where you
# want to store MFCC features.
featdir=$PWD/exp/kaldi_wsj_feats
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
  steps/make_mfcc.sh data/$x exp/make_mfcc/$x $featdir/mfcc 10
  rm -rf data-fbank/$x; 
  cp -r data/$x data-fbank/$x
  steps/make_fbank.sh data-fbank/$x data/$x exp/make_fbank/$x $featdir/fbank 10
done



######################################################
### PREPARE THE SUBSETS                            ###
######################################################

mkdir data/train_si84
for x in feats.scp text utt2spk wav.scp; do
  head -7138 data/train_si284/$x > data/train_si84/$x
done
scripts/utt2spk_to_spk2utt.pl data/train_si84/utt2spk > data/train_si84/spk2utt || exit 1;
scripts/filter_scp.pl data/train_si84/spk2utt data/train_si284/spk2gender > data/train_si84/spk2gender || exit 1;

# Now make subset with the shortest 2k utterances from si-84.
scripts/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
scripts/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;

# subset si84 on fbank feats
mkdir data-fbank/train_si84
for x in feats.scp text utt2spk wav.scp; do
  head -7138 data-fbank/train_si284/$x > data-fbank/train_si84/$x
done
scripts/utt2spk_to_spk2utt.pl data-fbank/train_si84/utt2spk > data-fbank/train_si84/spk2utt || exit 1;
scripts/filter_scp.pl data-fbank/train_si84/spk2utt data-fbank/train_si284/spk2gender > data-fbank/train_si84/spk2gender || exit 1;



######################################################
### TRAIN THE BASELINE GMM SYSTEMS                 ###
######################################################

# ### A : mono0a ### #
# --num-jobs queue option will be supplied to all alignment
# and training scripts.  Note: you have to supply the same num-jobs
# to the alignment and training scripts, as the archives are split
# up in this way.
#
# Train mono0a with 2k shortest segments of si84 data.
steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang exp/mono0a || exit 1;
# Decode mono0a
(
$mkgraph_cmd exp/mono0a/_mkgraph.log scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a exp/mono0a/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_dev93 exp/mono0a/decode_tgpr_dev93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92 || exit 1;
)&
# Align si84 with mono0a -> mono1a
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/mono0a exp/mono0a_ali_si84
# Align si84half with mono0a -> tri1
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84_half data/lang exp/mono0a exp/mono0a_ali_si84_half || exit 1;



# ### B : mono1a ### #
# Train better monophone system on si84 data, 
# (more accurate monophone alignments will be useful for MLP training)
# Train mono1a with si84 data.
steps/train_mono_ali.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/mono0a_ali_si84 exp/mono1a
# Decode mono1a
(
$mkgraph_cmd exp/mono1a/_mkgraph.log scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono1a exp/mono1a/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono1a/graph_tgpr data/test_dev93 exp/mono1a/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono1a/graph_tgpr data/test_eval92 exp/mono1a/decode_tgpr_eval92
)&
# Align si84 with mono1a
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/mono1a exp/mono1a_ali_si84
# Align si284 with mono1a
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si284 data/lang exp/mono1a exp/mono1a_ali_si284
# Align dev93 with mono1a
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/test_dev93 data/lang exp/mono1a exp/mono1a_ali_dev93



# ### C : tri1 ### #
# Proceed in training the triphone baselines:
# Train tri1, which is deltas + delta-deltas, on half of si84 data.
steps/train_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
    2000 10000 data/train_si84_half data/lang exp/mono0a_ali_si84_half exp/tri1 || exit 1;
# Build graph
wait; # the langage models are built on background
# or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.
(
$mkgraph_cmd exp/tri1/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr || exit 1;
# Decode dev93, eval92
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri1/graph_tgpr data/test_dev93 exp/tri1/decode_tgpr_dev93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri1/graph_tgpr data/test_eval92 exp/tri1/decode_tgpr_eval92 || exit 1;
)&
# Align si84 with tri1
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84 || exit 1;



# ### D : tri2a ### #
# Train tri2a, which is deltas + delta-deltas, on si84 data.
# This system will produce alignment for MLP training,
# optionally change number of leaves/PDFs.
numleaves=2500
{
  dir=exp/tri2a
  ali=exp/tri1_ali
  # Train
  steps/train_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    $numleaves 15000 data/train_si84 data/lang ${ali}_si84 $dir || exit 1;
  # Decode
  (
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
  )&
  # Align si84 with tri2a
  (steps/align_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    --use-graphs data/train_si84 data/lang $dir ${dir}_ali_si84)&
  # Align si284 with tri2a
  (steps/align_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    data/train_si284 data/lang $dir ${dir}_ali_si284)&
  # Align dev93 with tri2a
  (steps/align_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    data/test_dev93 data/lang $dir ${dir}_ali_dev93)&
}
wait



# ### E : tri2b ### #
# Train tri2b, which is LDA+MLLT, on si84 data.
# The alignments from this setup have been found worse
# for the MLP training than tri2a.
numleaves=2500
{
  dir=exp/tri2b
  ali=exp/tri1_ali
  # Train
  steps/train_lda_mllt.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 data/train_si84 data/lang ${ali}_si84 $dir || exit 1;
  # Decode
  (
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  )&
  # Align si84 with tri2b
  (steps/align_lda_mllt.sh  --num-jobs 10 --cmd "$train_cmd" \
    --use-graphs data/train_si84 data/lang $dir ${dir}_ali_si84)&
  # Align dev93 with tri2b
  (steps/align_lda_mllt.sh  --num-jobs 10 --cmd "$train_cmd" \
    data/test_dev93 data/lang $dir ${dir}_ali_dev93)&
}
wait 



# ### F : tri3b ### #
# Train tri2b, which is LDA+MLLT+SAT, on si84 data.
numleaves=2500
{
  dir=exp/tri3b
  ali=exp/tri2b_ali
  # Train
  steps/train_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
    $numleaves 15000 data/train_si84 data/lang ${ali}_si84 $dir || exit 1;
  # Decode
  (
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  )&

  # Align si84 with tri3b
  (steps/align_lda_mllt_sat.sh  --num-jobs 10 --cmd "$train_cmd" \
    --use-graphs data/train_si84 data/lang $dir ${dir}_ali_si84)&
  # Align dev93 with tri2b
  (steps/align_lda_mllt_sat.sh  --num-jobs 10 --cmd "$train_cmd" \
    data/test_dev93 data/lang $dir ${dir}_ali_dev93)&
}
wait 



