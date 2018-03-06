#!/bin/bash
#
# Test QATIP on IFN/ENIT database (set-s)

# if [ $# != 1 ]; then
#   echo "Usage: local/test_ienit.sh <feat-method>"
#   echo "feat-method: fsushi, fslytherin"
#   exit 1
# fi

featMethod=$1

oldLC=$LC_ALL

source path.sh
source cmd.sh

if [ "a" = "b" ]; then 

# Create data/test_ienit directory
if [ ! -d data/test_ienit ]
then
  echo "--> Create data/test_ienit"
  exit
  mkdir data/test_ienit
  cat ../corpus/ienit/text.test | sort -k1 > tmp.sorted
  cat tmp.sorted | cut -d' ' -f1 > data/test_ienit/uttids
  export LC_ALL=$oldLC
  cat tmp.sorted | cut -d' ' -f2- | python3 local/remove_diacritics.py | python3 local/replace_arabic_punctuation.py | python3 local/replace_brackets.py | tr '+' '\\' | tr '=' '\\' | sed 's/\xA0/X/g' | sed 's/\x00\xA0/X/g' | sed 's/\xC2\xA0/X/g' | sed 's/\s\+/ /g' | sed 's/ \+$//' | sed 's/^ \+$//' | paste -d' ' data/test_ienit/uttids - > data/test_ienit/text
  cat data/test_ienit/text | awk '{print $0" ("$1")"}' | cut -d' ' -f2- > data/test_ienit/text.trn
  export LC_ALL=C
  cat data/test_ienit/text.trn | local/word2char-trn.sh > data/test_ienit/text.clevel.trn
  cat data/test_ienit/uttids | sed 's/^\([a-z]\+\)-/\1 /i' | awk '{print "../corpus/"$1"/normalized/"$1"-"$2".png"}' | xargs -n 1 realpath > data/test_ienit/img.flist
  paste -d' ' data/test_ienit/uttids data/test_ienit/img.flist > data/test_ienit/img.scp
  cat data/test_ienit/uttids | sed 's/^\([^_]\+\)_/\1 /' | awk '{if (NF < 2) print $1" "$1; else print $1"_"$2" "$1}' > data/test_ienit/utt2spk
  utils/utt2spk_to_spk2utt.pl data/test_ienit/utt2spk > data/test_ienit/spk2utt
fi

# Create data/local_ienit directory
if [ ! -d data/local_ienit ]
then
  echo "--> Create data/local_ienit"
  mkdir data/local_ienit
  cp -r data/local/dict data/local_ienit 
  rm data/local_ienit/dict/lexicon*
  mkdir data/local_ienit/lm
  export LC_ALL=$oldLC
  cat ../corpus/ienit/text.train | cut -d' ' -f2 | sort -u > data/local_ienit/lm/vocab
  cat data/local_ienit/lm/vocab | python3 local/get_atb_pronun.py > data/test_ienit/words2latin
  ../../tools/srilm/bin/i686-m64/ngram-count -text data/local_ienit/lm/vocab -order 2 -lm data/local_ienit/lm/vocab.lm
  gzip data/local_ienit/lm/vocab.lm 
  #cat data/test_ienit/words2latin | python3 local/map_to_rareA.py data/local/phonemeset > data/local_ienit/dict/lexicon.txt
  cat data/test_ienit/words2latin | python3 local/map_to_rareA.py data/local/phonemeset | sed 's/\s\+la[BM]\{1\}\s\+conn\s\+a[meha]\{1\}E/ laLE/g' | python3 local/add_ligature_variants.py ../../config/ligatures > data/local_ienit/dict/lexicon.txt
  echo "<unk> rareA" >> data/local_ienit/dict/lexicon.txt
  echo "!SIL sil" >> data/local_ienit/dict/lexicon.txt
  export LC_ALL=C
fi

# Create data/lang_ienit directory
if [ ! -d data/lang_ienit ]
then
  echo "--> Create data/lang_ienit"
  echo '<unk>' > data/lang_ienit_nolm/oov.txt
  utils/prepare_lang.sh --topo-file data/lang/topo --position-dependent-phones false data/local_ienit/dict "<unk>" data/local_ienit/lang data/lang_ienit_nolm
  utils/format_lm_sri.sh --srilm-opts "-order 2" data/lang_ienit_nolm data/local_ienit/lm/vocab.lm.gz data/local_ienit/dict/lexicon.txt data/lang_ienit
fi

# Feature extraction
if [ ! -f "data/feats/fsushi_test_ienit.ark,t" ]
then
  echo "--> Feature extraction"
  ../../tools/prepocressor/prepocressor -inputFile data/test_ienit/img.flist -nThreads 1 -outputPath 'no' -pipeline "grayscale|convertToFloat|normalize -newMax 1|featExtract -winWidth 3 -winShift 2 -featRawCellHeight 1 -featRawCellWidth 1 -featRawCellShift 1 -kaldiFile data/feats/fsushi_test_ienit.ark,t |devNull"
  ../../tools/prepocressor/prepocressor -inputFile data/test_ienit/img.flist -nThreads 1 -outputPath 'no' -pipeline "grayscale|convertToFloat|normalize -newMax 1|featExtract -extractors snake -winWidth 3 -winShift 2 -kaldiFile data/feats/fslytherin_test_ienit.ark,t |devNull"
fi
if [ ! -f "data/feats/fsushi.nothinning_test_ienit.ark,t" ]
then
  echo "--> Feature extraction"
  ../../tools/prepocressor/prepocressor -inputFile data/test_ienit/img.flist -nThreads 1 -outputPath 'no' -pipeline "grayscale|convertToFloat|normalize -newMax 1|featExtract -winWidth 3 -winShift 2 -featRawCellHeight 1 -featRawCellWidth 1 -featRawCellShift 1 -kaldiFile data/feats/fsushi.nothinning_test_ienit.ark,t |devNull"
  ../../tools/prepocressor/prepocressor -inputFile data/test_ienit/img.flist -nThreads 1 -outputPath 'no' -pipeline "grayscale|convertToFloat|normalize -newMax 1|featExtract -extractors snake -winWidth 3 -winShift 2 -kaldiFile data/feats/fslytherin.nothinning_test_ienit.ark,t |devNull"
fi

# Feature preparation
if [ ! -f "data/test_ienit/feats.scp" ]
then
  if [ ! -f "data/feats/${featMethod}_test_ienit.ark,t" ]
  then
    echo "Feature extraction method not found."
    exit 1
  fi
  rm -r data/test_ienit/split*
  copy-feats ark,t:data/feats/${featMethod}_test_ienit.ark,t ark,scp:data/feats/${featMethod}_test_ienit.ark,data/test_ienit/feats.scp
  steps/compute_cmvn_stats.sh --fake data/test_ienit data/cmvn_ienit data/cmvn_ienit
  utils/validate_data_dir.sh --no-wav data/test_ienit/
  split_data.sh data/test_ienit 8 || exit 1;
fi
fi

if [ "a" = "b" ]
then
# mono_pregdl decoding graph
if [ -f exp/mono_pregdl/final.mdl ]
then
  utils/mkgraph.sh --mono data/lang_pregdl exp/mono_pregdl exp/mono_pregdl/graph_ienit
fi

# decode mono_pregdl [LDA+MLLT]
if [ -f exp/mono_pregdl/final.mdl ]
then
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
     exp/mono_pregdl/graph_ienit data/test exp/mono_pregdl/decode_ienit
  local/score.sh --cmd "$cmd" data/test exp/mono_pregdl/graph_ienit exp/mono_pregdl/decode_ienit
  # local/add_sclite.sh exp/mono_pregdl/decode_ienit/scoring/ data/lang_pregdl/words.txt data/test/text.trn data/test_ienit/text.clevel.trn
fi
fi

if [ "a" = "b" ]
then
# tri2b_pregdl decoding graph
if [ -f exp/tri2b_pregdl/final.mdl ]
then
  utils/mkgraph.sh data/lang_pregdl exp/tri2b_pregdl exp/tri2b_pregdl/graph_ienit
fi

# decode tri2b_pregdl [LDA+MLLT]
if [ -f exp/tri2b_pregdl/final.mdl ]
then
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
     exp/tri2b_pregdl/graph_ienit data/test exp/tri2b_pregdl/decode_ienit
  local/score.sh --cmd "$cmd" data/test exp/tri2b_pregdl/graph_ienit exp/tri2b_pregdl/decode_ienit
  # local/add_sclite.sh exp/mono_pregdl/decode_ienit/scoring/ data/lang_pregdl/words.txt data/test/text.trn data/test_ienit/text.clevel.trn
fi
fi

if [ "a" = "b" ]
then
# tri2b_mmi decoding graph
if [ -f exp/tri2b_mmi/final.mdl ]
then
  utils/mkgraph.sh data/lang exp/tri2b_mmi exp/tri2b_mmi/graph_ienit
fi

# decode tri2b_mmi [LDA+MLLT]
if [ -f exp/tri2b_mmi/final.mdl ]
then
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
     exp/tri2b_mmi/graph_ienit data/test exp/tri2b_mmi/decode_ienit
  local/score.sh --cmd "$cmd" data/test exp/tri2b_mmi/graph_ienit exp/tri2b_mmi/decode_ienit
  # local/add_sclite.sh exp/tri2b_mmi/decode_ienit/scoring/ data/lang/words.txt data/test/text.trn data/test/text.clevel.trn
fi
fi

if [ "a" != "b" ]
then
# tri3b_mmi decoding graph
if [ -f exp/tri3b_mmi/final.mdl ]
then
  utils/mkgraph.sh data/lang exp/tri3b_mmi exp/tri3b_mmi/graph_ienit
fi

# decode tri3b_mmi [LDA+MLLT]
if [ -f exp/tri3b_mmi/final.mdl ]
then
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
     exp/tri3b_mmi/graph_ienit data/test exp/tri3b_mmi/decode_ienit
  local/score.sh --cmd "$cmd" data/test exp/tri3b_mmi/graph_ienit exp/tri3b_mmi/decode_ienit
  # local/add_sclite.sh exp/tri3b_mmi/decode_ienit/scoring/ data/lang/words.txt data/test/text.trn data/test/text.clevel.trn
fi
fi

if [ "a" = "b" ]
then
# tri3b decoding graph
if [ -f exp/tri3b/final.mdl ]
then
  utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph_ienit
fi

# decode tri3b [LDA+MLLT]
if [ -f exp/tri3b/final.mdl ]
then
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
     exp/tri3b/graph_ienit data/test exp/tri3b/decode_ienit
  local/score.sh --cmd "$cmd" data/test exp/tri3b/graph_ienit exp/tri3b/decode_ienit
  # local/add_sclite.sh exp/tri3b/decode_ienit/scoring/ data/lang/words.txt data/test/text.trn data/test/text.clevel.trn
fi
fi



if [ "a" = "b" ]
then
# tri2b decoding graph
if [ -f exp/tri2b/final.mdl ] && [ ! -d "exp/tri2b/graph_ienit" ]
then
  utils/mkgraph.sh data/lang_ienit exp/tri2b exp/tri2b/graph_ienit
fi

# decode tri2b [LDA+MLLT]
if [ -f exp/tri2b/final.mdl ] && [ ! -d "exp/tri2b/decode_ienit" ]
then
  steps/decode.sh --config conf/decode.config --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b/decode_ienit
  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b/decode_ienit
  local/add_sclite.sh exp/tri2b/decode_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

# decode tri2b with MMI
if [ -f exp/tri2b_mmi/final.mdl ] && [ ! -d "exp/tri2b_mmi/decode_it4_ienit" ]
then
  steps/decode.sh --config conf/decode.config --iter 4 --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b_mmi/decode_it4_ienit
  steps/decode.sh --config conf/decode.config --iter 3 --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b_mmi/decode_it3_ienit

  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b_mmi/decode_it4_ienit
  local/add_sclite.sh exp/tri2b_mmi/decode_it4_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn

  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b_mmi/decode_it3_ienit
  local/add_sclite.sh exp/tri2b_mmi/decode_it3_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

# decode tri2b with MMI and silence boosting
if [ -f exp/tri2b_mmi_b0.05/final.mdl ] && [ ! -d "exp/tri2b_mmi_b0.05/decode_it4_ienit" ]
then
  steps/decode.sh --config conf/decode.config --iter 4 --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b_mmi_b0.05/decode_it4_ienit
  steps/decode.sh --config conf/decode.config --iter 3 --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b_mmi_b0.05/decode_it3_ienit

  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b_mmi_b0.05/decode_it4_ienit
  local/add_sclite.sh exp/tri2b_mmi_b0.05/decode_it4_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn

  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b_mmi_b0.05/decode_it3_ienit
  local/add_sclite.sh exp/tri2b_mmi_b0.05/decode_it3_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

# decode tri2b with MPE (discriminative)
if [ -f exp/tri2b_mpe/final.mdl ] && [ ! -d "exp/tri2b_mpe/decode_it4_ienit" ]
then
  steps/decode.sh --config conf/decode.config --iter 4 --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b_mpe/decode_it4_ienit
  steps/decode.sh --config conf/decode.config --iter 3 --nj 8 --cmd "$decode_cmd" \
     exp/tri2b/graph_ienit data/test_ienit exp/tri2b_mpe/decode_it3_ienit

  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b_mpe/decode_it4_ienit
  local/add_sclite.sh exp/tri2b_mpe/decode_it4_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn

  local/score.sh --cmd "$cmd" data/test_ienit exp/tri2b/graph_ienit exp/tri2b_mpe/decode_it3_ienit
  local/add_sclite.sh exp/tri2b_mpe/decode_it3_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

# tri3b decoding graph
if [ -f exp/tri3b/final.mdl ] && [ ! -d "exp/tri3b/graph_ienit" ]
then
  utils/mkgraph.sh data/lang_ienit exp/tri3b exp/tri3b/graph_ienit
fi

# decode tri3b 
if [ -f exp/tri3b/final.mdl ] && [ ! -d "exp/tri3b/decode_ienit" ]
then
  steps/decode_fmllr.sh --config conf/decode.config --nj 8 --cmd "$decode_cmd" \
    exp/tri3b/graph_ienit data/test_ienit exp/tri3b/decode_ienit
  local/score.sh --cmd "$cmd" data/test_ienit exp/tri3b/graph_ienit exp/tri3b/decode_ienit
  local/add_sclite.sh exp/tri3b/decode_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

# decode nnet4d
if [ -f exp/nnet4d/final.mdl ] && [ ! -d "exp/nnet4d/decode_ienit" ]
then
  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri3b/decode_ienit \
    exp/tri3b/graph_ienit data/test_ienit exp/nnet4d/decode_ienit  
  local/add_sclite.sh exp/nnet4d/decode_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

# decode nnet5d_mpe
if [ -f exp/nnet5d_mpe/final.mdl ] && [ ! -d "exp/nnet5d_mpe/decode_it1_ienit" ]
then
  for epoch in 1 2 3 4; do
    steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 --iter epoch${epoch} \
      --transform-dir exp/tri3b/decode_ienit \
      exp/tri3b/graph_ienit data/test_ienit exp/nnet5d_mpe/decode_it${epoch}_ienit  
    local/add_sclite.sh exp/nnet5d_mpe/decode_it${epoch}_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
  done
fi

# decode lstm4f
if [ -f exp/lstm4f/final.nnet ] && [ ! -d "exp/lstm4f/decode_ienit" ]
then
  steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    exp/tri3b/graph_ienit data/test_ienit exp/lstm4f/decode_ienit || exit 1;
  local/add_sclite.sh exp/lstm4f/decode_ienit/scoring/ data/lang_ienit/words.txt data/test_ienit/text.trn data/test_ienit/text.clevel.trn
fi

exit


# 5d
dir="exp/nnet5d_mpe"
for epoch in 1 2 3 4; do
  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch${epoch} \
    --transform-dir exp/tri3b/decode_infnt \
    exp/tri3b/graph_infnt data/test_infnt $dir/decode_epoch${epoch}_infnt &
  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch${epoch} \
    --transform-dir exp/tri3b/decode_ug_infnt \
    exp/tri3b/graph_ug_infnt data/test_infnt $dir/decode_ug_epoch${epoch}_infnt 
done
wait
./local/add_sclite.sh exp/nnet5d_mpe/decode_epoch1_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/nnet5d_mpe/decode_epoch2_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/nnet5d_mpe/decode_epoch3_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/nnet5d_mpe/decode_epoch4_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn



dir="exp/nnet4d"
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_infnt \
  exp/tri3b/graph_infnt data/test_infnt $dir/decode_infnt  
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_ug_infnt \
  exp/tri3b/graph_ug_infnt data/test_infnt $dir/decode_ug_infnt
./local/add_sclite.sh exp/nnet4d/decode_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/nnet4d/decode_ug_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn


steps/decode.sh --config conf/decode.config --iter 4 --nj 8 --cmd "$decode_cmd" \
   exp/tri2b/graph_infnt data/test_infnt exp/tri2b_mpe/decode_it4_infnt
steps/decode.sh --config conf/decode.config --iter 3 --nj 8 --cmd "$decode_cmd" \
   exp/tri2b/graph_infnt data/test_infnt exp/tri2b_mpe/decode_it3_infnt

local/score.sh --cmd "$cmd" data/test_infnt exp/tri2b/graph_infnt exp/tri2b_mmi/decode_it3_infnt
./local/add_sclite.sh exp/tri2b_mmi/decode_it3_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
local/score.sh --cmd "$cmd" data/test_infnt exp/tri2b/graph_infnt exp/tri2b_mmi/decode_it4_infnt
./local/add_sclite.sh exp/tri2b_mmi/decode_it4_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn


# Do the same with boosting.
steps/decode.sh --config conf/decode.config --iter 4 --nj 8 --cmd "$decode_cmd" \
   exp/tri2b/graph_infnt data/test_infnt exp/tri2b_mmi_b0.05/decode_it4_infnt
steps/decode.sh --config conf/decode.config --iter 3 --nj 8 --cmd "$decode_cmd" \
   exp/tri2b/graph_infnt data/test_infnt exp/tri2b_mmi_b0.05/decode_it3_infnt

local/score.sh --cmd "$cmd" data/test_infnt exp/tri2b/graph_infnt exp/tri2b_mmi_b0.05/decode_it3_infnt
./local/add_sclite.sh exp/tri2b_mmi_b0.05/decode_it3_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
local/score.sh --cmd "$cmd" data/test_infnt exp/tri2b/graph_infnt exp/tri2b_mmi_b0.05/decode_it4_infnt
./local/add_sclite.sh exp/tri2b_mmi_b0.05/decode_it4_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn

# Do MPE.

local/score.sh --cmd "$cmd" data/test_infnt exp/tri2b/graph_infnt exp/tri2b_mpe/decode_it3_infnt
./local/add_sclite.sh exp/tri2b_mpe/decode_it3_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
local/score.sh --cmd "$cmd" data/test_infnt exp/tri2b/graph_infnt exp/tri2b_mpe/decode_it4_infnt
./local/add_sclite.sh exp/tri2b_mpe/decode_it4_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn

## Do LDA+MLLT+SAT, and decode.
utils/mkgraph.sh data/lang_infnt exp/tri3b exp/tri3b/graph_infnt
steps/decode_fmllr.sh --config conf/decode.config --nj 8 --cmd "$decode_cmd" \
  exp/tri3b/graph_infnt data/test_infnt exp/tri3b/decode_infnt


cp -rT data/lang_infnt data/lang_ug_infnt
(
 utils/mkgraph.sh data/lang_ug_infnt exp/tri3b exp/tri3b/graph_ug_infnt
 steps/decode_fmllr.sh --config conf/decode.config --nj 8 --cmd "$decode_cmd" \
   exp/tri3b/graph_ug_infnt data/test_infnt exp/tri3b/decode_ug_infnt
)
./local/add_sclite.sh exp/tri3b/decode_ug_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn



./local/add_sclite.sh exp/tri3b/decode_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/tri3b/decode.si_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn


# # We have now added a script that will help you find portions of your data that
# # has bad transcripts, so you can filter it out.  Below we demonstrate how to
# # run this script.
# steps/cleanup/find_bad_utts.sh --nj 20 --cmd "$train_cmd" data/train data/lang \
#   exp/tri3b_ali exp/tri3b_cleanup 
# # The following command will show you some of the hardest-to-align utterances in the data.
# head  exp/tri3b_cleanup/all_info.sorted.txt 

## MMI on top of tri3b (i.e. LDA+MLLT+SAT+MMI)

steps/decode_fmllr.sh --config conf/decode.config --nj 8 --cmd "$decode_cmd" \
  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
   exp/tri3b/graph_infnt data/test_infnt exp/tri3b_mmi/decode_infnt

# Do a decoding that uses the exp/tri3b/decode directory to get transforms from.
steps/decode.sh --config conf/decode.config --nj 8 --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/tri3b/graph_infnt data/test exp/tri3b_mmi/decode2_infnt

./local/add_sclite.sh exp/tri3b_mmi/decode_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/tri3b_mmi/decode.si_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/tri3b_mmi/decode2_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn

# demonstration scripts for online decoding.
# local/online/run_gmm.sh
# local/online/run_nnet2.sh
# local/online/run_baseline.sh
# Note: for online decoding with pitch, look at local/run_pitch.sh, 
# which calls local/online/run_gmm_pitch.sh

#
# local/run_nnet2.sh
# local/online/run_nnet2_baseline.sh



#first, train UBM for fMMI experiments.

#for iter in 3 4 5 6 7 8; do
# steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
#   --transform-dir exp/tri3b/decode_infnt  exp/tri3b/graph_infnt data/test_infnt exp/tri3b_fmmi_b/decode_it${iter}_infnt &
#done


# 4d
dir="exp/nnet4d"
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_infnt \
  exp/tri3b/graph_infnt data/test_infnt $dir/decode_infnt  
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_ug_infnt \
  exp/tri3b/graph_ug_infnt data/test_infnt $dir/decode_ug_infnt
./local/add_sclite.sh exp/nnet4d/decode_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/nnet4d/decode_ug_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn

# 4d2
dir="exp/nnet4d2"
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_infnt \
  exp/tri3b/graph_infnt data/test_infnt $dir/decode_infnt  
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_ug_infnt \
  exp/tri3b/graph_ug_infnt data/test_infnt $dir/decode_ug_infnt
./local/add_sclite.sh exp/nnet4d2/decode_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn
./local/add_sclite.sh exp/nnet4d2/decode_ug_infnt/scoring/ data/lang_infnt/words.txt data/test_infnt/text.trn data/test_infnt/text.clevel.pi.trn

# Demo of "raw fMLLR"
# local/run_raw_fmllr.sh


# You don't have to run all 3 of the below, e.g. you can just run the run_sgmm2.sh
#local/run_sgmm.sh
#local/run_sgmm2.sh
#local/run_sgmm2x.sh

# The following script depends on local/run_raw_fmllr.sh having been run.
#
# local/run_nnet2.sh

# Karel's neural net recipe.                                                                                                                                        
# local/nnet/run_dnn.sh                                                                                                                                                  

# Karel's CNN recipe.
# local/nnet/run_cnn.sh
fi