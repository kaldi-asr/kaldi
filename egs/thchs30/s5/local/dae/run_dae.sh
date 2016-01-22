#!/bin/bash

#dnn dae training

dwntest=false
stdtest=false
stage=0
nj=8

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)
. utils/parse_options.sh || exit 1;


#generate noisy data. We focuse on the 0db condition.
#For training set, generate noisy data with SNR mean=0, variance=10, with three noise types mixed together.  
#For dev, generate noisy data with SNR mean=0, variance=0, with three niose types mixed together
#For test, either use the standard test set (stdtest=true) or randomly generated data (stdtest=false)
#The standard noisy test data are availabe at the challenge website (by setting dwntest=true):
#http://data.cslt.org/thchs30/challenges/asr.html

if [ $stage =  0 ]; then
   # generate training data
   noise_scp=local/dae/noise.scp
   noise_prior="0.0,10.0,10.0,10.0" #define noise type to sample. [S_clean, S_white, S_car, S_cafe]
   noise_level=0 #0db condition
   sigma0=10 #some random in SNR
   seed=32
   verbose=0

   addnoise_opt="--noise-level $noise_level --sigma0 $sigma0 --seed $seed --verbose $verbose --noise-prior $noise_prior --noise-src $noise_scp --wavdir NULL"

   rm -rf data/dae/train && mkdir -p data/dae/train || exit 1
   cp -L data/fbank/train/{wav.scp,spk2utt,utt2spk,text} data/dae/train || exit 1
   local/dae/make_fbank.sh --nj $nj --cmd "$train_cmd"  \
   --addnoise-opt """$addnoise_opt"""  \
   data/dae/train exp/dae/gendata fbank/dae/train || exit 1
   steps/compute_cmvn_stats.sh data/dae/train exp/dae/cmvn \
   fbank/dae/train/_cmvn || exit 1


   # genreate dev data. Just the 0db condition is produced.  Multiple noise types mixed together.
   sigma0=0 #no random in SNR
   addnoise_opt="--noise-level $noise_level --sigma0 $sigma0 --seed $seed --verbose $verbose --noise-prior $noise_prior --noise-src $noise_scp --wavdir NULL"
   rm -rf data/dae/dev/0db && mkdir -p data/dae/dev/0db && \
   cp -L data/fbank/dev/{wav.scp,spk2utt,utt2spk,text} data/dae/dev/0db || exit 1
   local/dae/make_fbank.sh --nj $nj --cmd "$train_cmd"  \
   --addnoise-opt """$addnoise_opt"""  \
   data/dae/dev/0db exp/dae/gendata fbank/dae/dev/0db || exit 1
   steps/compute_cmvn_stats.sh data/dae/dev/0db exp/dae/cmvn \
   fbank/dae/dev/0db/_cmvn || exit 1


   # generate test data. Note that if you want to compare with the standard results, set stdtest=true
   if [ $stdtest = true ]; then
     #download noisy wav if use the standard test data
     echo "using standard test data"
     if [ $dwntest = true ];then
       echo "downloading the noisy test data from Tsinghua Univ..."
       echo "this may be slow for some connections, and is not very stable."
       echo "you may want to try alternative mirror sites."
       echo "check http://data.cslt.org/thchs30/challenges/asr.html"
       (
        mkdir -p wav && cd wav && \
        wget http://data.cslt.org/thchs30-openslr/test.0db.tgz || exit 1
        tar xvf test.0db.tgz || exit 1
       )
     fi
     #generate fbank
     for x in car white cafe; do
       echo "producing fbanks for $x"
       mkdir -p data/dae/test/0db/$x && \
       cp -L data/fbank/test/{spk2utt,utt2spk,text} data/dae/test/0db/$x && \
       awk '{print $1 " wav/test-noise/0db/'$x'/"$1".wav"}' data/fbank/test/wav.scp > data/dae/test/0db/$x/wav.scp || exit 1
       steps/make_fbank.sh --nj $nj --cmd "$train_cmd"  \
        data/dae/test/0db/$x exp/dae/gendata fbank/dae/test/0db/$x || exit 1
     done

   else
     #generate test data randomly
     echo "generating noisy test data randomly.."

     for x in car white cafe; do
       case $x in
         car)
            noise_prior="0.0,0.0,10.0,0.0" 
            ;;
         white)
            noise_prior="0.0,10.0,0.0,0.0" 
            ;;
         cafe)
            noise_prior="0.0,0.0,0.0,10.0" 
            ;;
       esac

       addnoise_opt="--noise-level $noise_level --sigma0 $sigma0 --seed $seed --verbose $verbose --noise-prior $noise_prior --noise-src $noise_scp --wavdir NULL"
       rm -rf data/dae/test/0db/$x && mkdir -p data/dae/test/0db/$x && \
       cp -L data/fbank/test/{wav.scp,spk2utt,utt2spk,text} data/dae/test/0db/$x || exit 1
       local/dae/make_fbank.sh --nj $nj --cmd "$train_cmd"  \
         --addnoise-opt """$addnoise_opt"""  \
         data/dae/test/0db/$x exp/dae/gendata fbank/dae/test/0db/$x || exit 1

     done
   fi

   for x in car white cafe; do
     echo "generating cmvn for test data.."
     steps/compute_cmvn_stats.sh data/dae/test/0db/$x exp/dae/cmvn \
      fbank/dae/test/0db/$x/_cmvn || exit 1
     echo "producing phone data dir.."
     cp -R data/dae/test/0db/$x data/dae/test/0db/$x.ph && cp data/test/phone.txt data/dae/test/0db/$x.ph/text || exit 1
   done
fi

if [ $stage -le 1 ]; then
  #train dnn dae using data with mixed noise
  $cuda_cmd exp/tri4b_dnn_dae/log/train_nnet.log \
  local/dae/train.sh --hid-layers 2 --hid-dim 1200 \
  --cmvn-opts "--norm-vars=false"  --splice 10 \
  --learn-rate 0.0001 \
  --train_tool_opts "--objective-function=mse" \
  --copy_feats false \
  data/dae/train data/dae/dev/0db data/lang data/fbank/train data/fbank/dev exp/tri4b_dnn_dae || exit 1;
  nnet-concat exp/tri4b_dnn_dae/final.feature_transform exp/tri4b_dnn_dae/final.nnet \
  exp/tri4b_dnn_mpe/final.feature_transform exp/tri4b_dnn_dae/dae.nnet  || exit 1
  
fi

#decoding 
if [ $stage -le 2 ]; then
   for x in car white cafe; do
     (
       #decode word 
       steps/nnet/decode.sh --cmd "$decode_cmd" --nj $nj \
        --srcdir exp/tri4b_dnn_mpe \
       exp/tri4b/graph.word data/dae/test/0db/$x exp/tri4b_dnn_mpe/decode_word_0db/$x|| exit 1;

       steps/nnet/decode.sh --cmd "$decode_cmd" --nj $nj \
        --srcdir exp/tri4b_dnn_mpe --feature-transform exp/tri4b_dnn_dae/dae.nnet \
       exp/tri4b/graph.word data/dae/test/0db/$x exp/tri4b_dnn_dae/decode_word_0db/$x|| exit 1;
       #decode phone
       steps/nnet/decode.sh --cmd "$decode_cmd" --nj $nj \
        --srcdir exp/tri4b_dnn_mpe \
       exp/tri4b/graph.phone data/dae/test/0db/$x.ph exp/tri4b_dnn_mpe/decode_phone_0db/$x|| exit 1;

       steps/nnet/decode.sh --cmd "$decode_cmd" --nj $nj \
        --srcdir exp/tri4b_dnn_mpe --feature-transform exp/tri4b_dnn_dae/dae.nnet \
       exp/tri4b/graph.phone data/dae/test/0db/$x.ph exp/tri4b_dnn_dae/decode_phone_0db/$x|| exit 1;
    ) &
   done
fi

