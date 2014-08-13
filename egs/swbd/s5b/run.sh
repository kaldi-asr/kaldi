#!/bin/bash

# This recipe is based on the run_edin.sh recipe, by Arnab Ghoshal,
# in the s5/ directory.
# This is supposed to be the "new" version of the switchboard recipe,
# after the s5/ one became a bit messy.  It is not 100% checked-through yet.

#exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. cmd.sh
. path.sh
set -e # exit on error

# Prepare Switchboard data. This command can also take a second optional argument
# which specifies the directory to Switchboard documentations. Specifically, if
# this argument is given, the script will look for the conv.tab file and correct
# speaker IDs to the actual speaker personal identification numbers released in
# the documentations. The documentations can be found here: 
# https://catalog.ldc.upenn.edu/docs/LDC97S62/
# Note: if you are using this link, make sure you rename conv_tab.csv to conv.tab
# after downloading.
# Usage: local/swbd1_data_prep.sh /path/to/SWBD [/path/to/SWBD_docs]
local/swbd1_data_prep.sh /export/corpora3/LDC/LDC97S62
# local/swbd1_data_prep.sh /home/dpovey/data/LDC97S62
# local/swbd1_data_prep.sh /data/corpora0/LDC97S62
# local/swbd1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2
# local/swbd1_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/switchboard1

local/swbd1_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

# Now train the language models. We are using SRILM and interpolating with an
# LM trained on the Fisher transcripts (part 2 disk is currently missing; so 
# only part 1 transcripts ~700hr are used)

# If you have the Fisher data, you can set this "fisher_opt" variable.
fisher_opt="--fisher /export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/"
#fisher_opt="--fisher /home/dpovey/data/LDC2004T19/fe_03_p1_tran/"
#fisher_opt="--fisher /data/corpora0/LDC2004T19/fe_03_p1_tran/"
# edinburgh:
# fisher_opt="--fisher /exports/work/inf_hcrc_cstr_general/corpora/fisher/transcripts"
# brno:
# fisher_opt="--fisher /mnt/matylda2/data/FISHER/fe_03_p1_tran" # BUT
local/swbd1_train_lms.sh $fisher_opt \
  data/local/train/text data/local/dict/lexicon.txt data/local/lm
# We don't really need all these options for SRILM, since the LM training script
# does some of the same processings (e.g. -subset -tolower)
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
LM=data/local/lm/sw1.o3g.kn.gz
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  data/lang $LM data/local/dict/lexicon.txt data/lang_sw1_tg

LM=data/local/lm/sw1_fsh.o3g.kn.gz
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  data/lang $LM data/local/dict/lexicon.txt data/lang_sw1_fsh_tg

# For some funny reason we are still using IRSTLM for doing LM pruning :)
export PATH=$PATH:../../../tools/irstlm/bin/
prune-lm --threshold=1e-7 data/local/lm/sw1_fsh.o3g.kn.gz /dev/stdout \
  | gzip -c > data/local/lm/sw1_fsh.o3g.pr1-7.kn.gz || exit 1
LM=data/local/lm/sw1_fsh.o3g.pr1-7.kn.gz
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  data/lang $LM data/local/dict/lexicon.txt data/lang_sw1_fsh_tgpr


# Data preparation and formatting for eval2000 (note: the "text" file
# is not very much preprocessed; for actual WER reporting we'll use
# sclite.

# local/eval2000_data_prep.sh /data/corpora0/LDC2002S09/hub5e_00 /data/corpora0/LDC2002T43
# local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/2000_hub5_eng_eval_tr
# local/eval2000_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/hub5/2000 /exports/work/inf_hcrc_cstr_general/corpora/switchboard/hub5/2000/transcr
# local/eval2000_data_prep.sh /home/dpovey/data/LDC2002S09/hub5e_00 /home/dpovey/data/LDC2002T43
local/eval2000_data_prep.sh /export/corpora2/LDC/LDC2002S09/hub5e_00 /export/corpora2/LDC/LDC2002T43

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfcc

steps/make_mfcc.sh --compress true --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir 

# Remove the small number of utterances that couldn't be extracted for some 
# reason (e.g. too short; no such file).
utils/fix_data_dir.sh data/train 

# Create MFCCs for the eval set
steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_mfcc/eval2000 $mfccdir
steps/compute_cmvn_stats.sh data/eval2000 exp/make_mfcc/eval2000 $mfccdir
utils/fix_data_dir.sh data/eval2000  # remove segments with problems

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.
utils/subset_data_dir.sh --first data/train 4000 data/train_dev # 5hr 6min
n=$[`cat data/train/segments | wc -l` - 4000]
utils/subset_data_dir.sh --last data/train $n data/train_nodev

# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' data/local/train/segments


# Now-- there are 260k utterances (313hr 23min), and we want to start the 
# monophone training on relatively short utterances (easier to align), but not 
# only the shortest ones (mostly uh-huh).  So take the 100k shortest ones;
# remove most of the repeated utterances (these are the uh-huh type ones), and 
# then take 10k random utterances from those (about 4hr 40mins)

utils/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort
local/remove_dup_utts.sh 10 data/train_100kshort data/train_100kshort_nodup
utils/subset_data_dir.sh data/train_100kshort_nodup 10000 data/train_10k_nodup

# Take the first 30k utterances (about 1/8th of the data)
utils/subset_data_dir.sh --first data/train_nodev 30000 data/train_30k
local/remove_dup_utts.sh 200 data/train_30k data/train_30k_nodup  # 33hr

# Take the first 100k utterances (just under half the data); we'll use
# this for later stages of training.
utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup  # 110hr

# Finally, the full training set:
local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  # 286hr

## Starting basic training on MFCC features
steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono 

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup data/lang exp/mono exp/mono_ali 

steps/train_deltas.sh --cmd "$train_cmd" \
  3200 30000 data/train_30k_nodup data/lang exp/mono_ali exp/tri1 

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri1/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri1 $graph_dir
    steps/decode_si.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/eval2000 exp/tri1/decode_eval2000_sw1_${lm_suffix}
  ) &
done

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup data/lang exp/tri1 exp/tri1_ali 

steps/train_deltas.sh --cmd "$train_cmd" \
  3200 30000 data/train_30k_nodup data/lang exp/tri1_ali exp/tri2 


for lm_suffix in tg fsh_tgpr; do
  (
    # The previous mkgraph might be writing to this file.  If the previous mkgraph
    # is not running, you can remove this loop and this mkgraph will create it.
    while [ ! -s data/lang_sw1_${lm_suffix}/tmp/CLG_3_1.fst ]; do sleep 60; done
    sleep 20; # in case still writing.
    graph_dir=exp/tri2/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri2 $graph_dir
    steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/eval2000 exp/tri2/decode_eval2000_sw1_${lm_suffix}
  ) &
done

# From now, we start building a bigger system (on train_100k_nodup, which has 
# 110hrs of data). We start with the LDA+MLLT system
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri2 exp/tri2_ali_100k_nodup 

# Train tri3b, which is LDA+MLLT, on 100k_nodup data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  5500 90000 data/train_100k_nodup data/lang exp/tri2_ali_100k_nodup exp/tri3b 

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri3b/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri3b $graph_dir
    steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/eval2000 exp/tri3b/decode_eval2000_sw1_${lm_suffix}
  ) &
done

# Train tri4a, which is LDA+MLLT+SAT, on 100k_nodup data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri3b exp/tri3b_ali_100k_nodup 


steps/train_sat.sh  --cmd "$train_cmd" \
  5500 90000 data/train_100k_nodup data/lang exp/tri3b_ali_100k_nodup \
   exp/tri4a 

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri4a/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri4a $graph_dir
    steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/eval2000 exp/tri4a/decode_eval2000_sw1_${lm_suffix}
  ) &
done


#local/run_resegment.sh

# Now train a LDA+MLLT+SAT model on the entire training data (train_nodup; 
# 286 hours)
# Train tri4b, which is LDA+MLLT+SAT, on train_nodup data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri3b exp/tri3b_ali_nodup 


steps/train_sat.sh  --cmd "$train_cmd" \
  11500 200000 data/train_nodup data/lang exp/tri3b_ali_nodup exp/tri4b

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri4b $graph_dir
    steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       $graph_dir data/eval2000 exp/tri4b/decode_eval2000_sw1_${lm_suffix}
    steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       $graph_dir data/train_dev exp/tri4b/decode_train_dev_sw1_${lm_suffix}
  ) &
done
wait
steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_fsh_tgpr data/lang_sw1_fsh_tg data/eval2000 \
  exp/tri4b/decode_eval2000_sw1_fsh_tgpr exp/tri4b/decode_eval2000_sw1_fsh_tg.3 || exit 1


# MMI training starting from the LDA+MLLT+SAT systems on both the 
# train_100k_nodup (110hr) and train_nodup (286hr) sets
steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_ali_100k_nodup || exit 1

steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4b exp/tri4b_ali_nodup || exit 1

steps/make_denlats.sh --nj 50 --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri4a_ali_100k_nodup \
  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_denlats_100k_nodup \
  

steps/make_denlats.sh --nj 100 --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri4b_ali_nodup \
  data/train_nodup data/lang exp/tri4b exp/tri4b_denlats_nodup 

# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
num_mmi_iters=4
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --num-iters $num_mmi_iters \
  data/train_100k_nodup data/lang exp/tri4a_{ali,denlats}_100k_nodup \
  exp/tri4a_mmi_b0.1 

steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --num-iters $num_mmi_iters \
  data/train_nodup data/lang exp/tri4b_{ali,denlats}_nodup \
  exp/tri4b_mmi_b0.1 

for iter in 1 2 3 4; do
  for lm_suffix in tg fsh_tgpr; do
    (
      graph_dir=exp/tri4a/graph_sw1_${lm_suffix}
      decode_dir=exp/tri4a_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_${lm_suffix}
    steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    --iter $iter --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
            $graph_dir data/eval2000 $decode_dir
  ) &
  done
done

for iter in 1 2 3 4; do
  for lm_suffix in tg fsh_tgpr; do
    (
      graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
      decode_dir=exp/tri4b_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_${lm_suffix}
      steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    --iter $iter --transform-dir exp/tri4b/decode_eval2000_sw1_${lm_suffix} \
    $graph_dir data/eval2000 $decode_dir   
  ) &
  done
done

#TODO(arnab): add lmrescore here
# ./steps/lmrescore.sh --mode 3 --cmd "$highmem_cmd" data/lang_sw1_fsh_tgpr data/lang_sw1_fsh_tg data/eval2000 exp/tri3a/decode_eval2000_sw1_fsh_tgpr exp/tri3a/decode_eval2000_sw1_fsh_tg.3 &

# Now do fMMI+MMI training
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 50 --cmd "$train_cmd" \
  700 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri4a_dubm

steps/train_diag_ubm.sh --silence-weight 0.5 --nj 100 --cmd "$train_cmd" \
  700 data/train_nodup data/lang exp/tri4b_ali_nodup exp/tri4b_dubm

steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri4a_dubm \
  exp/tri4a_denlats_100k_nodup exp/tri4a_fmmi_b0.1 

steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4b_ali_nodup exp/tri4b_dubm \
  exp/tri4b_denlats_nodup exp/tri4b_fmmi_b0.1  

for iter in 4 5 6 7 8; do
  for lm_suffix in tg fsh_tgpr; do
    (
      graph_dir=exp/tri4a/graph_sw1_${lm_suffix}
      decode_dir=exp/tri4a_fmmi_b0.1/decode_eval2000_it${iter}_sw1_${lm_suffix}
      steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
	--transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
	--config conf/decode.config $graph_dir data/eval2000 $decode_dir
    ) &
  done
done

for iter in 4 5 6 7 8; do
  for lm_suffix in tg fsh_tgpr; do
    (
      graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
      decode_dir=exp/tri4b_fmmi_b0.1/decode_eval2000_it${iter}_sw1_${lm_suffix}
      steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
	--transform-dir exp/tri4b/decode_eval2000_sw1_${lm_suffix} \
	--config conf/decode.config $graph_dir data/eval2000 $decode_dir
    ) &
  done
done


# local/run_sgmm2.sh

# demonstration script for raw-fMLLR.  You should probably ignore this.
# local/run_raw_fmllr.sh


# # Karel's DNN recipe on top of fMLLR features
# local/run_dnn.sh


# # Dan's nnet recipe
# # you might want to look into that script and run parts,
# # rather than just running the whole thing.
# local/run_nnet2.sh

# # Dan's nnet recipe with online decoding.
# local/online/run_nnet2.sh
# local/online/run_nnet2_baseline.sh


# # getting results (see RESULTS file)
# for x in 1 2 3a 3b 4a; do grep 'Percent Total Error' exp/tri$x/decode_eval2000_sw1_tg/score_*/eval2000.ctm.filt.dtl | sort -k5 -g | head -1; done

