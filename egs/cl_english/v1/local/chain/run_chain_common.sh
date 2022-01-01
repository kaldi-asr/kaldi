#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

# The only path modification this script does is to append _hires to test set directory names
# for extracting the ivectos.

set -euo pipefail

stage=0
lores_train_data_dir=data/train_sp
train_data_dir=data/train_sp_hires
gmm=exp/tri3b
ali_lats_dir=exp/tri3b_lats_train
lang=data/lang
lang_chain=data/lang
tree_dir=exp/chain/tree_sp
leaves=4500
test_sets="test_ldc test_sp_oc"
nj=10
tree_opts="--context-width=2 --central-position=1"
exp=exp

online_cmvn_iextractor=false
use_ivector=true
extractor=
ivector_dim=100
nnet3_affix=
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ -z $extractor ]; then
  extractor=$exp/nnet3${nnet3_affix}/extractor
fi

train_set=$(basename $train_data_dir)
echo "$0: Highres train data dir: $train_data_dir"
echo "$0: Lowres train data dir: ${lores_train_data_dir}"

for f in ${lores_train_data_dir}/feats.scp $train_data_dir/feats.scp $gmm/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [[ $stage -le 5 ]] && [[ ! -z $gmm ]]; then
  if [[ -f $ali_lats_dir/lat.1.gz ]] && [[ $ali_lats_dir/lat.1.gz -nt $gmm/final.mdl ]]; then
    printf "\n$0: The lattices seem to be there and up to date wrt to gmm model. Skipping\n\n"
  else
    echo "$0: Generating alignments and lattices for "
    if [ -f $gmm/trans.1 ]; then # It's fmllr
      steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd"  \
                                --generate-ali-from-lats true \
                                $lores_train_data_dir \
                                $lang $gmm $ali_lats_dir || exit 1;
    else
      local/align_si_lats.sh --nj $nj --cmd "$train_cmd" \
                                  --generate-ali-from-lats true \
                                  $lores_train_data_dir $lang $gmm $ali_lats_dir
    fi
    rm $ali_lats_dir/fsts.*.gz 2>/dev/null || true # save space
  fi
  sleep 2
fi


if [ $stage -le 6 ]; then
  echo "$0: Creating lang directory $lang_chain with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang_chain ]; then
    if [ $lang_chain/L.fst -nt $lang/L.fst ]; then
      echo "$0: $lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: $lang_chain already exists and seems to be older than data/lang..."
    fi
  else
    cp -r $lang $lang_chain
    silphonelist=$(cat $lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang_chain/topo
  fi
  sleep 2
fi

if [ $stage -le 7 ]; then
  echo "$0: Buidling the tree..."
  if [[ -f $tree_dir/final.mdl ]]; then
    printf "\n$0: $tree_dir/final.mdl already exists. Skipping.\n\n"
  elif [ -z "$tree_dir" ]; then
    printf "\n$0: Tree dir is empty. Skipping tree stage.\n\n"
  else
      steps/nnet3/chain/build_tree.sh \
        --frame-subsampling-factor 3 \
        --context-opts "$tree_opts" \
        --cmd "$train_cmd" $leaves ${lores_train_data_dir} \
        $lang_chain $ali_lats_dir $tree_dir
  fi
  sleep 2
fi

if ! $use_ivector; then
  echo "$0: ## Not doing ivectors ##"
  sleep 2
  exit 0;
fi


if [[ -f $extractor/final.ie ]] && [[ $stage -le 9  ]]; then
    echo ""
    echo "$0: There is already an ivector extractor trained. Skipping..."
    echo ""
else
    if [ $stage -le 8 ]; then
      echo "$0: computing a subset of data to train the diagonal UBM."
      # We'll use about a quarter of the data.
      mkdir -p $exp/nnet3${nnet3_affix}/diag_ubm
      temp_data_root=$exp/nnet3${nnet3_affix}/diag_ubm

      num_utts_total=$(wc -l <data/${train_set}/utt2spk)
      num_utts=$[$num_utts_total/4]
      utils/data/subset_data_dir.sh data/${train_set} \
         $num_utts ${temp_data_root}/${train_set}_subset

      echo "$0: computing a PCA transform from the hires data."
      steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
          --splice-opts "--left-context=3 --right-context=3" \
          --max-utts 10000 --subsample 2 \
           ${temp_data_root}/${train_set}_subset \
           $exp/nnet3${nnet3_affix}/pca_transform

      echo "$0: training the diagonal UBM."
      # Use 512 Gaussians in the UBM.
      steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $[$nj/2] \
        --num-frames 700000 \
        --num-threads 8 \
        ${temp_data_root}/${train_set}_subset 512 \
        $exp/nnet3${nnet3_affix}/pca_transform $exp/nnet3${nnet3_affix}/diag_ubm
    fi

    if [ $stage -le 9 ]; then
      # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
      # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
      # 100.
      echo "$0: training the iVector extractor"
      steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
         --num-threads 4 --num-processes 2 --ivector-dim $ivector_dim \
         --online-cmvn-iextractor $online_cmvn_iextractor \
         data/${train_set} $exp/nnet3${nnet3_affix}/diag_ubm \
         $extractor || exit 1;
    fi
    sleep 2
fi

if [ $stage -le 10 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  ivectordir=$exp/nnet3${nnet3_affix}/ivectors_${train_set}

  if [ -f $ivectordir/ivector_online.scp ]; then
      echo ""
      echo "iVectors already there for $train_set. Skipping. Check compatibility yourself!"
      echo ""
  else
      # having a larger number of speakers is helpful for generalization, and to
      # handle per-utterance decoding well (iVector starts at zero).
      temp_data_root=${ivectordir}
      utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
        data/${train_set} ${temp_data_root}/${train_set}_max2

      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        ${temp_data_root}/${train_set}_max2 \
        $extractor $ivectordir
  fi
  sleep 2
  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  echo "--- $0: $test_sets "
  for data in $test_sets; do
    odir=$exp/nnet3${nnet3_affix}/ivectors_${data}_hires
    if [ -f $odir/ivector_online.scp ]; then
        echo ""
        echo "iVectors already there for $data. Skipping. Check compatibility yourself!"
        echo ""
    else
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nspk \
                                                    data/${data}_hires $extractor $odir
    fi
  done
  sleep 2
fi

exit 0
