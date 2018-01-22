#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh


n=8 # parallel jobs


###### Bookmark: basic preparation ######

# corpus and trans directory
thchs=/nfs/public/materials/data/thchs30-openslr

# you can obtain the database by uncommting the following lines
# [ -d $thchs ] || mkdir -p $thchs  || exit 1
# echo "downloading THCHS30 at $thchs ..."
# local/download_and_untar.sh $thchs  http://www.openslr.org/resources/18 data_thchs30  || exit 1
# local/download_and_untar.sh $thchs  http://www.openslr.org/resources/18 resource      || exit 1
# local/download_and_untar.sh $thchs  http://www.openslr.org/resources/18 test-noise    || exit 1

# generate text, wav.scp, utt2pk, spk2utt
local/thchs-30_data_prep.sh $thchs/data_thchs30 || exit 1;


###### Bookmark: language preparation ######

# prepare lexicon.txt, extra_questions.txt, nonsilence_phones.txt, optional_silence.txt, silence_phones.txt
# build a large lexicon that invovles words in both the training and decoding
mkdir -p data/dict;
cp $thchs/resource/dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} data/dict && \
cat $thchs/resource/dict/lexicon.txt $thchs/data_thchs30/lm_word/lexicon.txt | \
grep -v '<s>' | grep -v '</s>' | sort -u > data/dict/lexicon.txt || exit 1;


###### Bookmark: language processing ######

# generate language stuff used for training
# also lexicon to L_disambig.fst for graph making in local/thchs-30_decode.sh
mkdir -p data/lang;
utils/prepare_lang.sh --position_dependent_phones false data/dict "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;

# format trained or provided language model to G.fst
# prepare things for graph making in local/thchs-30_decode.sh, not necessary for training
(
  mkdir -p data/graph;
  gzip -c $thchs/data_thchs30/lm_word/word.3gram.lm > data/graph/word.3gram.lm.gz || exit 1;
  utils/format_lm.sh data/lang data/graph/word.3gram.lm.gz $thchs/data_thchs30/lm_word/lexicon.txt data/graph/lang || exit 1;
)


###### Bookmark: feature extraction ######

# produce MFCC and Fbank features
rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,test} data/mfcc || exit 1;
rm -rf data/fbank && mkdir -p data/fbank && cp -r data/{train,test} data/fbank || exit 1;
for x in train test; do
  # make mfcc and fbank
  steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x || exit 1;
  steps/make_fbank.sh --nj $n --cmd "$train_cmd" data/fbank/$x || exit 1;
  # compute cmvn
  steps/compute_cmvn_stats.sh data/mfcc/$x || exit 1;
  steps/compute_cmvn_stats.sh data/fbank/$x || exit 1;
done


###### Bookmark: GMM-HMM training & decoding ######

# monophone
steps/train_mono.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono || exit 1;
# test monophone model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/mono data/mfcc &
# monophone ali
steps/align_si.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono exp/mono_ali || exit 1;

# triphone
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/mfcc/train data/lang exp/mono_ali exp/tri1 || exit 1;
# test tri1 model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/tri1 data/mfcc &
# triphone_ali
steps/align_si.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# lda_mllt
steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" 2500 15000 data/mfcc/train data/lang exp/tri1_ali exp/tri2b || exit 1;
# test tri2b model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/tri2b data/mfcc &
# lda_mllt_ali
steps/align_si.sh  --nj $n --cmd "$train_cmd" --use-graphs true data/mfcc/train data/lang exp/tri2b exp/tri2b_ali || exit 1;

# sat
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 data/mfcc/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
# test tri3b model
local/thchs-30_decode.sh --nj $n "steps/decode_fmllr.sh" exp/tri3b data/mfcc &
# sat_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

# quick
steps/train_quick.sh --cmd "$train_cmd" 4200 40000 data/mfcc/train data/lang exp/tri3b_ali exp/tri4b || exit 1;
# test tri4b model
local/thchs-30_decode.sh --nj $n "steps/decode_fmllr.sh" exp/tri4b data/mfcc &
# quick_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri4b exp/tri4b_ali || exit 1;


###### Bookmark: DNN training and decoding ######

# train tdnn model
tdnn_dir=exp/nnet3/tdnn
local/nnet3/run_tdnn.sh data/fbank/train exp/tri4b_ali $tdnn_dir || exit 1;

# decoding
graph_dir=exp/tri4b/graph_word # the same as gmm
steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" $graph_dir data/fbank/test $tdnn_dir/decode_test_word || exit 1;


###### Bookmark: discriminative training and decoding ######

# mmi training
criterion=mmi # mmi, mpfe or smbr
local/nnet3/run_tdnn_discriminative.sh --criterion $criterion $tdnn_dir data/fbank/train || exit 1;

# decoding
steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" $graph_dir data/fbank/test ${tdnn_dir}_$criterion/decode_test_word || exit 1;


exit 0;
