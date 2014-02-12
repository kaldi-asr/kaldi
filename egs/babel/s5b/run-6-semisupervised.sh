#!/bin/bash


#Run decoding of the
#DEVTRAIN
#UNTRANSCRIBED
#This yields approx 70 hours of data

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

set -u           #Fail on an undefined variable

target=`pwd`-unsupervised2
min_cf=0.7
max_cf=1.0
expand_boundaries=0.0

. ./utils/parse_options.sh

if [ ! -d $target ]; then
  echo ---------------------------------------------------------------------
  echo "Creating directory $target for the semi-supervised trained system"
  echo ---------------------------------------------------------------------
  mkdir -p $target
  cp *.sh $target
  for x in steps utils local conf lang.conf; do
    [ ! -s $x ] && echo "No such file or directory $x" && exit 1;
    if [ -L $x ]; then # if these are links
      cp -d $x $target # copy the link over.
    else # create a link to here.
      ln -s ../`basename $PWD`/$x $target
    fi
  done
fi

mkdir -p $target/data

for dir in raw_train_data raw_dev2h_data raw_dev10h_data; do
  ln -s `pwd`/data/$dir $target/data/ || true;
done
ln -s `pwd`/data/dev10h.uem $target/data
ln -s `pwd`/data/dev2h.uem $target/data
ln -s `pwd`/data/dev10h $target/data
ln -s `pwd`/data/dev2h $target/data
ln -s `pwd`/data/lang $target/data
ln -s `pwd`/data/local $target/data
ln -s `pwd`/data/train $target/data/train_supervised
ln -s `pwd`/data/srilm $target/data

#This, of course, should be done on the top of the UEM segmentation
for train_data_source in devtrain.uem semitrain.uem ; do

  tgt_datadir=data/train_${train_data_source}
  mkdir -p $target/$tgt_datadir

  local/ctm2segments.pl --min-cf $min_cf --max-cf $max_cf --cf-rule min \
    data/${train_data_source} exp/sgmm5_mmi_b0.1/decode_fmllr_${train_data_source}_it1/score_10/${train_data_source}.utt.ctm $target/$tgt_datadir

  cp data/${train_data_source}/wav.scp  $target/$tgt_datadir

  utils/utt2spk_to_spk2utt.pl $target/$tgt_datadir/utt2spk > $target/$tgt_datadir/spk2utt
  (cd $target
    if [ ! -f $tgt_datadir/.plp.done ]; then
      if [ "$use_pitch" = "false" ] ; then
        steps/make_plp.sh       --cmd "$train_cmd" --nj $train_nj ${tgt_datadir} \
          exp/make_features/train_${train_data_source} plp || exit 1
      else
        steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj ${tgt_datadir} \
          exp/make_features/train_${train_data_source} plp || exit 1
      fi
      utils/fix_data_dir.sh ${tgt_datadir}
      steps/compute_cmvn_stats.sh \
        ${tgt_datadir} exp/make_features/train_${train_data_source} plp
      # In case plp or pitch extraction failed on some utterances, delist them
      utils/fix_data_dir.sh ${tgt_datadir}
      touch ${tgt_datadir}/.plp.done
    fi
  )
done
#cp $target/data/train_supervised/text $target/data/train/text
utils/combine_data.sh $target/data/train $target/data/train_supervised $target/data/train_{devtrain.uem,semitrain.uem}
utils/fix_data_dir.sh $target/data/train

#This is to ensure the "correct" order of files so that the scripts won't try to redo something
touch $target/data/train/.plp.done
touch $target/data/local/lexicon.txt
touch $target/data/lang/L.fst
touch $target/data/srilm/lm.gz
touch $target/data/lang/G.fst

(
  mkdir -p $target/exp
  ln -sf `pwd`/exp/tri5 $target/exp/tri5
  
  cd $target
  . ./lang.conf
  if [ ! -f exp/tri5_ali/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting exp/tri5_ali on" `date`
    echo ---------------------------------------------------------------------
    steps/align_fmllr.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang exp/tri5 exp/tri5_ali || exit 1
    touch exp/tri5_ali/.done
  fi

  steps/nnet2/update_pnorm.sh --minibatch-size 128 \
    --num-jobs-nnet 8 --num-threads 16 --parallel-opts '-pe smp 16' \
    --cmd 'queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G' \
    data/train data/lang exp/tri5_ali /export/a13/jtrmal/babel/206-zulu-limitedLP-kpitch2/exp/tri6_nnet exp/tri6_nnet
  touch exp/tri6_nnet/.done 

 ./run-5-anydecode2.sh --skip-kws true



  #--steps/update_nnet_cpu.sh \
  #--  --mix-up "$dnn_mixup" \
  #--  --initial-learning-rate "$dnn_final_learning_rate" \
  #--  --final-learning-rate "$dnn_final_learning_rate" \
  #--  --num-hidden-layers "$dnn_num_hidden_layers" \
  #--  --num-parameters "$dnn_num_parameters" \
  #--  --num-jobs-nnet $dnn_num_jobs \
  #--  --cmd "$train_cmd" \
  #--  "${dnn_train_extra_opts[@]}" \
  #--  --num-epochs 1 \
  #--  --num-jobs-nnet 12 \
  #--  --cleanup false \
  #--  data/train data/lang exp/tri5_ali exp/tri6_nnet_2b

) || exit 1
