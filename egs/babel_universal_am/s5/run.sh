#!/bin/bash

###############################################################################
#                      Universal Acoustic Models
###############################################################################
#
# This script sets up data from the BABEL Languages to be used to train a
# universal acoustic models. By default, it leaves out 4 languages:
#
#  - 201_haitian: Many results numbers with which to compare for this language.
#  - 307_amharic: Left out to repeat LORELEI experiments.
#  - 107_vietnamese: Left out to test performance on a tonal language.
#  - 404_georgian: The final evaluation language from BABEL.
#
# which are used to test the trained universal acoustic models. The script
# consists of the following steps:
#   1. Prepare data directories
#   2. Standardize the lexicons
#   3. Training
###############################################################################
set -e
set -o pipefail
. ./path.sh
. ./lang.conf
. ./cmd.sh
. ./conf/common_vars.sh
###############################################################################
#                          PREPARE LANGUAGE DATA
###############################################################################
langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306
       401 402 403"

# Just a subset of the training languages for now.
# Decoding an unseen language takes more work to standardize the dictionary,
# and to replace missing phonemes.
decode_langs="105 206 304 403"

# Just for documentation and debugging mostly, but these should probably be
# organized differently to reflect important parts of the training script
# or just removed entirely.
# ------------------------------------------------------------------------
# stage 0 -- Setup Language Directories
# stage 1 -- Prepare Data
# stage 2 -- Combine Data
# stage 3 -- training
# stage 4 -- cleanup data and segmentation
# stage 5 -- Chain TDNN training
# stage 6 -- Prepare Decode data directory
# stage 7 -- Prepare Universal Dictionaries, LM
# stage 8 -- Make Decoding Graph
# stage 9 -- Prepare Decode acoustic data
# stage 10 -- Decode Chain

stage=0

. ./utils/parse_options.sh

set -x

# For each language create the data and also create the lexicon
# Save the current directory
cwd=$(utils/make_absolute.sh `pwd`)

if [ $stage -le 0 ]; then
  # For each language setup the language directories
  for l in ${langs}; do
    mkdir -p data/${l}
    cd data/${l}
    ln -sf ${cwd}/local .
    for f in ${cwd}/{utils,steps,conf}; do
      link=`make_absolute.sh $f`
      ln -sf $link .
    done
    conf_file=`find conf/lang -name "${l}-*limitedLP*.conf" -o -name "${l}-*LLP*.conf" | head -1`
    echo ${conf_file}
    cp $conf_file lang.conf

    # This line will likely not be when the lang.conf files are corrected.
    # It currently just fixes some paths on the CLSP grid that no longer exist.
    sed -i 's/export\/babel\/data\/splits/export\/babel\/data\/OtherLR-data\/splits/g' lang.conf
    cp ${cwd}/{cmd,path}.sh .
    cd $cwd
  done
fi

# For each language
for l in ${langs}; do
  cd data/${l}

  #############################################################################
  # Prepare the data (acoustic data and train directories)
  #############################################################################
  if [ $stage -le 1 ]; then
    ./local/prepare_data.sh

    ###########################################################################
    # Create dictionaries with split diphthongs and standardized tones
    ###########################################################################
    # In the lexicons provided by babel there are phonemes x_y, for which _y may
    # or may not best be considered as a tag on phoneme x. In Lithuanian, for
    # instance, there is a phoneme A_F for which _F or indicates failling tone.
    # This same linguistic feature is represented in other languages as a "tag"
    # (i.e. 判 pun3 p u: n _3), which means for the purposes of kaldi, that
    # those phonemes share a root in the clustering decision tree, and the tag
    # becomes an extra question. We may want to revisit this issue later.
    echo "Dictionary ${l}"
    dict=data/dict_universal
    diphthongs=${cwd}/universal_phone_maps/diphthongs/${l}
    tones=${cwd}/universal_phone_maps/tones/${l}

    mkdir -p $dict
    # Create silence lexicon
    echo -e "<silence>\tSIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" \
      > ${dict}/silence_lexicon.txt

    # Create non-silence lexicon
    grep -vFf ${dict}/silence_lexicon.txt data/local/lexicon.txt \
      > data/local/nonsilence_lexicon.txt

    # Create split diphthong and standarized tone lexicons for nonsilence words
    ./local/prepare_universal_lexicon.py \
      ${dict}/nonsilence_lexicon.txt data/local/nonsilence_lexicon.txt \
      $diphthongs $tones

    cat ${dict}/{,non}silence_lexicon.txt | sort > ${dict}/lexicon.txt

    # Prepare the rest of the dictionary directory
    # -----------------------------------------------
    # The local/prepare_dict.py script, which is basically the same as
    # prepare_unicode_lexicon.py used in the babel recipe to create the
    # graphemic lexicons, is better suited for working with kaldi formatted
    # lexicons and can be used for this task by only modifying optional input
    # arguments. If we could modify local/prepare_lexicon.pl to accomodate this
    # need it may be more intuitive.
    ./local/prepare_dict.py \
      --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}

    ###########################################################################
    # Prepend language ID to all utterances to disambiguate between speakers
    # of different languages sharing the same speaker id.
    #
    # The individual lang directories can be used for alignments, while a
    # combined directory will be used for training. This probably has minimal
    # impact on performance as only words repeated across languages will pose
    # problems and even amongst these, the main concern is the <hes> marker.
    ###########################################################################
    echo "Prepend ${l} to data dir"
    ./utils/copy_data_dir.sh --spk-prefix ${l} --utt-prefix ${l} \
      data/train data/train_${l}
  fi
  cd $cwd
done

###############################################################################
# Combine all langauge specific training directories and generate a single
# lang directory by combining all langauge specific dictionaries
###############################################################################
if [ $stage -le 2 ]; then
  train_dirs=""
  dict_dirs=""
  for l in ${langs}; do
    train_dirs="data/${l}/data/train_${l} ${train_dirs}"
    dict_dirs="data/${l}/data/dict_universal ${dict_dirs}"
  done

  ./utils/combine_data.sh data/train $train_dirs

  # This script was made to mimic the utils/combine_data.sh script, but instead
  # it merges the lexicons while reconciling the nonsilence_phones.txt,
  # silence_phones.txt, and extra_questions.txt by basically just calling
  # local/prepare_unicode_lexicon.py. As mentioned, it may be better to simply
  # modify an existing script to automatically create the dictionary dir from
  # a lexicon, rather than overuse the local/prepare_unicode_lexicon.py script.
  ./local/combine_lexicons.sh data/dict_universal $dict_dirs

  # Prepare lang directory
  ./utils/prepare_lang.sh --share-silence-phones true \
    data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal
fi



###############################################################################
#           Train the model through tri5 (like in babel recipe)
###############################################################################

# Currently, the full lang directory is used for alignments, but really each
# language specific directory should be used to get alignments for each
# language individually, and then combined to learn a shared tree over all
# languges. Language specific alignments will eliminate the problem of words
# shared across languages and consequently bad alignments. In practice, very
# few words are shared across languages, and when the are the pronunciations
# are often similar. The only real problem is for language specific hesitation
# markers <hes>.
#
# Training follows exactly the standard BABEL recipe. The number of Gaussians
# and leaves were previously tuned for a much larger dataset (10 langauges flp)
# which was about 700 hrs, instead of the 200 hrs here, but the same parameters
# are used here. These parameters could probably use some tweaking for this
# setup.


if [ $stage -le 3 ]; then
  if [ ! -f data/train_sub3/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting monophone training data in data/train_sub[123] on" `date`
    echo ---------------------------------------------------------------------
    numutt=`cat data/train/feats.scp | wc -l`;
    utils/subset_data_dir.sh data/train 5000 data/train_sub1
    if [ $numutt -gt 10000 ] ; then
      utils/subset_data_dir.sh data/train 10000 data/train_sub2
    else
      (cd data; ln -s train train_sub2 )
    fi
    if [ $numutt -gt 20000 ] ; then
      utils/subset_data_dir.sh data/train 20000 data/train_sub3
    else
      (cd data; ln -s train train_sub3 )
    fi

    touch data/train_sub3/.done
  fi


  if [ ! -f exp/mono/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting (small) monophone training in exp/mono on" `date`
    echo ---------------------------------------------------------------------
    steps/train_mono.sh \
      --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
      data/train_sub1 data/lang_universal exp/mono
    touch exp/mono/.done
  fi


  if [ ! -f exp/tri1/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting (small) triphone training in exp/tri1 on" `date`
    echo ---------------------------------------------------------------------
    steps/align_si.sh \
      --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
      data/train_sub2 data/lang_universal exp/mono exp/mono_ali_sub2

    steps/train_deltas.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
      data/train_sub2 data/lang_universal exp/mono_ali_sub2 exp/tri1

    touch exp/tri1/.done
  fi

  echo ---------------------------------------------------------------------
  echo "Starting (medium) triphone training in exp/tri2 on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -f exp/tri2/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
      data/train_sub3 data/lang_universal exp/tri1 exp/tri1_ali_sub3

    steps/train_deltas.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
      data/train_sub3 data/lang_universal exp/tri1_ali_sub3 exp/tri2

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train_sub3 data/lang_universal data/dict_universal \
      exp/tri2 data/dict_universal/dictp/tri2 data/dict_universal/langp/tri2 data/lang_universalp/tri2

    touch exp/tri2/.done
  fi

  echo ---------------------------------------------------------------------
  echo "Starting (full) triphone training in exp/tri3 on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -f exp/tri3/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri2 exp/tri2 exp/tri2_ali

    steps/train_deltas.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" \
      $numLeavesTri3 $numGaussTri3 data/train data/lang_universalp/tri2 exp/tri2_ali exp/tri3

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal/ \
      exp/tri3 data/dict_universal/dictp/tri3 data/dict_universal/langp/tri3 data/lang_universalp/tri3

    touch exp/tri3/.done
  fi


  echo ---------------------------------------------------------------------
  echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -f exp/tri4/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri3 exp/tri3 exp/tri3_ali

    steps/train_lda_mllt.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" \
      $numLeavesMLLT $numGaussMLLT data/train data/lang_universalp/tri3 exp/tri3_ali exp/tri4

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal \
      exp/tri4 data/dict_universal/dictp/tri4 data/dict_universal/langp/tri4 data/lang_universalp/tri4

    touch exp/tri4/.done
  fi

  echo ---------------------------------------------------------------------
  echo "Starting (SAT) triphone training in exp/tri5 on" `date`
  echo ---------------------------------------------------------------------

  if [ ! -f exp/tri5/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri4 exp/tri4 exp/tri4_ali

    steps/train_sat.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" \
      $numLeavesSAT $numGaussSAT data/train data/lang_universalp/tri4 exp/tri4_ali exp/tri5

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal \
      exp/tri5 data/dict_universal/dictp/tri5 data/dict_universal/langp/tri5 data/lang_universalp/tri5

    touch exp/tri5/.done
  fi

  if [ ! -f exp/tri5_ali/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting exp/tri5_ali on" `date`
    echo ---------------------------------------------------------------------
    steps/align_fmllr.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri5 exp/tri5 exp/tri5_ali

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal \
      exp/tri5_ali data/dict_universal/dictp/tri5_ali data/dict_universal/langp/tri5_ali data/lang_universalp/tri5_ali

    touch exp/tri5_ali/.done
  fi
fi


###############################################################################
#                          Data Cleanup
###############################################################################

# Issues:
#   1. There is an insufficient memory issue that arises in
#
#         steps/cleanup/make_biased_lm_graphs.sh
#
#      which I got around by using the -mem option in queue.pl and setting it
#      really high. This limits the number of jobs you can run and causes the
#      cleanup to be really slow. There is probably a better way around this.

if [ $stage -le 4 ]; then
  ./local/run_cleanup_segmentation.sh --langdir data/lang_universalp/tri5
fi

###############################################################################
#                                DNN Training
###############################################################################

if [ $stage -le 4 ]; then
  ./local/chain/run_tdnn.sh --langdir data/lang_universalp/tri5_ali
fi

###############################################################################
#============================== END OF TRAINING ===============================
###############################################################################

echo "Universal Acoustic Model Training finished." && \
echo "To decode, comment out these lines in run.sh (390-391)." && exit 0;




###############################################################################
#                                  Decoding
###############################################################################

# Preparing Decoding Data
# For each decoding language setup the language directories
for l in ${decode_langs}; do
  dict=data/dict
  langdir=data/lang_dict

  if [ $stage -le 5 ]; then
    mkdir -p data/${l}
    cd data/${l}
    ln -sf ${cwd}/local .
    for f in ${cwd}/{utils,steps,conf}; do
      link=`make_absolute.sh $f`
      ln -sf $link .
    done

    # Use the FLP training lexicons and text to train LM
    conf_file=`find conf/lang -name "${l}-*fullLP*.conf" -o -name "${l}-*FLP*.conf" | head -1`
    echo ${conf_file}
    cp $conf_file lang.conf

    # This line will likely not be when the lang.conf files are corrected.
    # It currently just fixes some paths on the CLSP grid that no longer exist.
    sed -i 's/export\/babel\/data\/splits/export\/babel\/data\/OtherLR-data\/splits/g' lang.conf
    cp ${cwd}/{cmd,path}.sh .
    ./local/prepare_data.sh --extract-feats false
  fi
  

  if [ $stage -le 6 ]; then 
    echo "------------------------------------------------------------"
    echo " Standardized Dictionaries and check Lang directories for"
    echo " compatibility with the Univeral Acoustic Models"
    echo "------------------------------------------------------------"
        
    mkdir -p ${dict}
    diphthongs=${cwd}/universal_phone_maps/diphthongs/${l}
    tones=${cwd}/universal_phone_maps/tones/${l}

    echo -e "<silence> SIL\n<unk> <oov>\n<noise> <sss>\n<v-noise> <vns>" \
      > ${dict}/silence_lexicon.txt
 
    # Create non-silence lexicon
    grep -vFf ${dict}/silence_lexicon.txt data/local/lexicon.txt \
      > data/local/nonsilence_lexicon.txt

    # Create split diphthong and standarized tone lexicons for nonsilence words
    ./local/prepare_universal_lexicon.py ${dict}/nonsilence_lexicon.txt \
      data/local/nonsilence_lexicon.txt $diphthongs $tones

    cat ${dict}/{,non}silence_lexicon.txt | sort > ${dict}/lexicon.txt
 
    # Create the rest of the dictionary    
    ./local/prepare_dict.py \
      --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}

    ./utils/prepare_lang.sh --share-silence-phones true \
      ${dict} "<unk>" ${dict}/tmp.lang ${langdir}
 
    ./local/phoneset_diff.sh ${langdir}/phones.txt \
      ${cwd}/data/lang_universalp/tri5_ali/phones.txt \
      > ${langdir}/missing_phones_map  
  fi 
  
  #############################################################################
  # MAP MISSING PHONEMES HERE
  #############################################################################
  
  # Resolve incompatibilities between diciontaries 
  if [ $stage -le 7 ]; then  
    ./local/convert_dict.sh ${dict}_universal \
      ${langdir}_universal \
      ${dict} ${cwd}/data/dict_universal \
      ${cwd}/data/lang_universalp/tri5_ali \
      ${langdir}/missing_phones_map

    ###########################################################################
    # Train the LM For the Decoding Language
    ###########################################################################
    
    ./local/train_lms_srilm.sh --oov-symbol "<unk>" \
                               --train-text data/train/text \
                               --words-file ${langdir}_universal/words.txt \
                               data data/srilm

    ./local/arpa2G.sh data/srilm/lm.gz ${langdir}_universal ${langdir}_universal
  fi
 
  # Make Decoding Graph 
  if [ $stage -le 8 ]; then
    ./utils/mkgraph.sh --self-loop-scale 1.0 ${langdir}_universal \
      ${cwd}/exp/chain_cleaned/tdnn_sp_bi \
      ${cwd}/exp/chain_cleaned/tdnn_sp_bi/graph_${l}
  fi

  # Prepare Acoustic Data
  if [ $stage -le 9 ]; then
    . ./lang.conf
    if [ ! -d data/raw_dev10h_data ]; then
      echo ---------------------------------------------------------------------
      echo "Subsetting the DEV10H set"
      echo ---------------------------------------------------------------------
      local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./data/raw_dev10h_data || exit 1
    fi

    mkdir -p data/dev10h.pem
    dev10h_data_dir=`utils/make_absolute.sh ./data/raw_dev10h_data`

    local/prepare_acoustic_training_data.pl --fragmentMarkers \-\*\~  \
      dev10h_data_dir data/dev10h.pem > data/dev10h.pem/skipped_utts.log || exit 1
    
    local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev10h.pem

    # Make plp + pitch features
    steps/make_plp_pitch.sh --nj 32 --cmd "$train_cmd" data/dev10h.pem exp/make_mfcc/dev10h.pem mfcc
    steps/compute_cmvn_stats.sh data/dev10h.pem exp/make_mfcc/dev10h.pem mfcc
  
    # Make hires mfcc features (for ivector extraction)
    utils/copy_data_dir.sh data/dev10h.pem data/dev10h.pem_hires
    steps/make_mfcc.sh --nj 32 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/dev10h.pem_hires exp/make_hires/dev10h.pem mfcc_hires;
    steps/compute_cmvn_stats.sh data/dev10h.pem_hires exp/make_hires/dev10h.pem mfcc_hires;
    utils/fix_data_dir.sh data/dev10h.pem_hires;

    # Make mfcc + pitch features (for nnet decoding)
    utils/copy_data_dir.sh data/dev10h.pem data/dev10h.pem_pitch_hires
    steps/make_mfcc_pitch.sh --nj 32 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/dev10h.pem_pitch_hires exp/make_hires/dev10h.pem_pitch mfcc_pitch_hires;
    steps/compute_cmvn_stats.sh data/dev10h.pem_pitch_hires exp/make_hires/dev10h.pem_pitch mfcc_pitch_hires;
    utils/fix_data_dir.sh data/dev10h.pem_pitch_hires;
  
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 32 \
      data/dev10h.pem_hires ${cwd}/exp/nnet3_cleaned/extractor/ ${cwd}/exp/nnet3_cleaned/ivectors_${l}_dev10h.pem/ || exit 1;
  
  fi
  
  # Decode
  if [ $stage -le 10 ]; then
    # Assign 100 / num_decode_langs nj per lang
    num_langs=`echo $decode_langs | wc -w`
    my_decode_nj=$((100 / $num_langs))

    (
      cd ${cwd};
      ./steps/nnet3/decode.sh --skip-scoring false \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $my_decode_nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3_cleaned/ivectors_${l}_dev10h.pem \
          exp/chain_cleaned/tdnn_sp_bi/graph_${l} \
          data/${l}/data/dev10h.pem_pitch_hires \
          exp/chain_cleaned/tdnn_sp_bi/decode_${l}_dev10h.pem
    ) &
  
  fi
done

