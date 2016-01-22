#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

H=`pwd`

thchs=/work3/zxw/thchs30/data_thchs30
# corpus and trans directory
# you can obtain the database by uncommting the following lines
# ( cd `dirname $thchs`
#     echo "downloading THCHS30 at $PWD ..."
#     wget http://www.openslr.org/resources/18/data_thchs30.tgz
#     tar xvf data_thchs30.tgz
# )
n=8

# data preparation 
# generate text, wav.scp, utt2pk, spk2utt

local/thchs-30_data_prep.sh $H $thchs || exit 1;

#produce MFCC features 
rm -rf data/mfcc && mkdir -p data/mfcc &&  cp -R data/{train,dev,test,test.ph} data/mfcc || exit 1;
for x in train dev test; do
   #make  mfcc 
   steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x exp/make_mfcc/$x mfcc/$x || exit 1;
   #compute cmvn
   steps/compute_cmvn_stats.sh data/mfcc/$x exp/mfcc_cmvn/$x mfcc/$x/_cmvn || exit 1;
done
#copy feats and cmvn to test.ph, avoid duplicated mfcc & cmvn 
cp data/mfcc/test/feats.scp data/mfcc/test.ph && cp data/mfcc/test/cmvn.scp data/mfcc/test.ph || exit 1;


# prepare language stuff
# build a large lexicon that invovles words in both the training and decoding. 
# The large lexioon is mainly for reasonable lattice generation in discriminative training
(
  echo "make word graph ..."
  cd $H; mkdir -p data/{dict,lang,graph} && \
  cp local/dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} data/dict && \
  cat local/dict/lexicon.txt $thchs/lm_word/lexcion.txt |grep -v '<s>'|grep -v '</s>' |sort -u > data/dict/lexicon.txt || exit 1;
  utils/prepare_lang.sh --position_dependent_phones false data/dict "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;
  gzip -c $thchs/lm_word/word.3gram.lm > data/graph/word.3gram.lm.gz || exit 1;
  utils/format_lm.sh data/lang data/graph/word.3gram.lm.gz $thchs/lm_word/lexcion.txt data/graph/lang || exit 1;
)

#make_phone_graph
(
  echo "make phone graph ..."
  cd $H; mkdir -p data/{dict.phone,graph.phone,lang.phone} && \
  cp local/dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} data/dict.phone  && \
  cat $thchs/lm_phone/lexcion.txt |grep -v '<eps>'|sort -u > data/dict.phone/lexicon.txt  && \
  echo "<SPOKEN_NOISE> sil " >> data/dict.phone/lexicon.txt  || exit 1;
  utils/prepare_lang.sh --position_dependent_phones false data/dict.phone "<SPOKEN_NOISE>" data/local/lang.phone data/lang.phone || exit 1;
  gzip -c $thchs/lm_phone/phone.3gram.lm > data/graph.phone/phone.3gram.lm.gz  || exit 1;
  utils/format_lm.sh data/lang.phone data/graph.phone/phone.3gram.lm.gz $thchs/lm_phone/lexcion.txt data/graph.phone/lang  || exit 1;
)

#monophone
steps/train_mono.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono || exit 1;
#test monophone model
local/thchs-30_decode.sh --mono true --nj $n "steps/decode.sh" exp/mono data/mfcc &

#monophone_ali
steps/align_si.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono exp/mono_ali || exit 1;

#triphone
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/mfcc/train data/lang exp/mono_ali exp/tri1 || exit 1;
#test tri1 model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/tri1 data/mfcc &

#triphone_ali
steps/align_si.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri1 exp/tri1_ali || exit 1;

#lda_mllt
steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" 2500 15000 data/mfcc/train data/lang exp/tri1_ali exp/tri2b || exit 1;
#test tri2b model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/tri2b data/mfcc &


#lda_mllt_ali
steps/align_si.sh  --nj $n --cmd "$train_cmd" --use-graphs true data/mfcc/train data/lang exp/tri2b exp/tri2b_ali || exit 1;

#sat
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 data/mfcc/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
#test tri3b model
local/thchs-30_decode.sh --nj $n "steps/decode_fmllr.sh" exp/tri3b data/mfcc &

#sat_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

#quick
steps/train_quick.sh --cmd "$train_cmd" 4200 40000 data/mfcc/train data/lang exp/tri3b_ali exp/tri4b || exit 1;
#test tri4b model
local/thchs-30_decode.sh --nj $n "steps/decode_fmllr.sh" exp/tri4b data/mfcc &

#quick_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri4b exp/tri4b_ali || exit 1;

#quick_ali_cv
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/dev data/lang exp/tri4b exp/tri4b_ali_cv || exit 1;

# train dnn model
local/nnet/run_dnn.sh --stage 0 --nj $n  exp/tri4b exp/tri4b_ali exp/tri4b_ali_cv || exit 1;  

# train dae model
local/dae/run_dae.sh --stage 0 || exit 1;
