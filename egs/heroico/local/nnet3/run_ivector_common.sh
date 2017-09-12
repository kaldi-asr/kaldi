#!/bin/bash

set -e -o pipefail

stage=0
nj=24
train_set=train
test_sets="native nonnative test"
gmm=tri3b
num_threads_ubm=24

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
    if [ ! -f $f ]; then
	echo "$0: expected file $f to exist"
	exit 1
    fi
done

if [ $stage -le 2 ] && [ -f data/${train_set}_sp_hires/feats.scp ]; then
    echo "$0: data/${train_set}_sp_hires/feats.scp already exists."
fi

if [ $stage -le 1 ]; then
    echo "$0: preparing directory for speed-perturbed data"
    utils/data/perturb_data_dir_speed_3way.sh \
	data/${train_set} \
	data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
    echo "$0: creating high-resolution MFCC features"

    mfccdir=data/${train_set}_sp_hires/data
    for datadir in ${train_set}_sp ${test_sets}; do
	utils/copy_data_dir.sh \
	    data/$datadir \
	    data/${datadir}_hires
    done

    # do volume-perturbation on the training data prior to extracting hires
    # features; this helps make trained nnets more invariant to test data volume.
    utils/data/perturb_data_dir_volume.sh \
	data/${train_set}_sp_hires

    for datadir in ${train_set}_sp ${test_sets}; do
	steps/make_mfcc.sh \
	    --nj $nj \
	    --mfcc-config conf/mfcc_hires.conf \
	    --cmd "$train_cmd" \
	    data/${datadir}_hires

	steps/compute_cmvn_stats.sh \
	    data/${datadir}_hires
	utils/fix_data_dir.sh \
	    data/${datadir}_hires
    done
fi

if [ $stage -le 3 ]; then
    echo "$0: selecting segments of hires training data that were also present in the"
    echo " ... original training data."

    temp_data_root=exp/nnet3/tri5
    mkdir -p $temp_data_root

    utils/data/subset_data_dir.sh \
	--utt-list data/${train_set}/feats.scp \
	data/${train_set}_sp_hires \
	$temp_data_root/${train_set}_hires

    n1=$(wc -l <data/${train_set}/feats.scp)
    n2=$(wc -l <$temp_data_root/${train_set}_hires/feats.scp)
    if [ $n1 != $n1 ]; then
	echo "$0: warning: number of feats $n1 != $n2, if these are very different it could be bad."
	sleep 5
    fi
    echo "$0: training a system on the hires data for its LDA+MLLT transform, in order to produce the diagonal GMM."
    if [ -e exp/nnet3/tri5/final.mdl ]; then
	echo "$0: exp/nnet3/tri5/final.mdl already exists: "
    fi

    steps/train_lda_mllt.sh \
	--cmd "$train_cmd" \
	--num-iters 7 \
	--mllt-iters "2 4 6" \
	--splice-opts "--left-context=3 --right-context=3" \
	3000 \
	10000 \
	$temp_data_root/${train_set}_hires \
	data/lang \
	$gmm_dir \
	exp/nnet3/tri5
fi

if [ $stage -le 4 ]; then
    echo "$0: computing a subset of data to train the diagonal UBM."

    mkdir -p exp/nnet3/diag_ubm
  temp_data_root=exp/nnet3/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
      $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh \
      --cmd "$train_cmd" \
      --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_sp_hires_subset \
    512 \
    exp/nnet3/tri5 \
    exp/nnet3/diag_ubm
fi

if [ $stage -le 5 ]; then
    # Train the iVector extractor.
    echo "$0: training the iVector extractor"
    steps/online/nnet2/train_ivector_extractor.sh \
	--cmd "$train_cmd" \
	--nj 10 \
	data/${train_set}_sp_hires \
	exp/nnet3/diag_ubm \
	exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
    ivectordir=exp/nnet3/ivectors_${train_set}_sp_hires
    #  extract iVectors on the speed-perturbed training data .
    #  pairs  utterances into twos,
    #  treats each  pair as one speaker;
    #  gives more diversity in iVectors.
    #  extracted 'online' (they vary within the utterance).
    #  larger number of speakers  helps  generalization,
    # handle per-utterance decoding well
    #  iVector starts at zero
    # beginning of each pseudo-speaker.

    temp_data_root=${ivectordir}
    utils/data/modify_speaker_info.sh \
	--utts-per-spk-max 2 \
	data/${train_set}_sp_hires \
	${temp_data_root}/${train_set}_sp_hires_max2

    steps/online/nnet2/extract_ivectors_online.sh \
	--cmd "$train_cmd" \
	--nj $nj \
	${temp_data_root}/${train_set}_sp_hires_max2 \
	exp/nnet3/extractor \
	$ivectordir

    # Also extract iVectors for the test data,
    # do not need  speed perturbation (sp).
    for data in ${test_sets}; do
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh \
	--cmd "$train_cmd" \
	--nj "${nspk}" \
	data/${data}_hires \
	exp/nnet3/extractor \
	exp/nnet3/ivectors_${data}_hires
    done
fi

if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 8 ]; then
    echo "$0: $feats already exists "
fi

if [ $stage -le 7 ]; then
    echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    data/${train_set} data/${train_set}_sp
fi

if [ $stage -le 8 ]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data (needed for alignments)"
  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 9 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir
fi


exit 0;
