#!/usr/bin/env bash
#set -e
# this script is based on local/online/run_nnet2_comman.sh
# but it operates on corrupted training/dev/test data sets


stage=1
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
num_data_reps=3
ali_dir=exp/
db_string="'air' 'rwcp' 'rvb2014'" # RIR dbs to be used in the experiment
                                      # only dbs used for ASpIRE submission system have been used here
RIR_home=db/RIR_databases/ # parent directory of the RIR databases files
download_rirs=true # download the RIR databases from the urls or assume they are present in the RIR_home directory

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# check if the required tools are present
local/multi_condition/check_version.sh || exit 1;

mkdir -p exp/nnet2_multicondition
if [ $stage -le 1 ]; then
  # prepare the impulse responses
  local/multi_condition/prepare_impulses_noises.sh --log-dir exp/make_reverb/log \
    --db-string "$db_string" \
    --download-rirs $download_rirs \
    --RIR-home $RIR_home \
    data/impulses_noises || exit 1;
    
  # Generate the rir_list and noise_list for the reverberate_data_dir.py to corrupt the data
  # this script just assumes air rwcp rvb2014 databases
  python local/multi_condition/aspire_prep_rir_noise_list.py data/impulses_noises data/impulses_noises/info

  # corrupt the fisher data to generate multi-condition data
  for data_dir in train dev test; do
    if [ "$data_dir" == "train" ]; then
      num_reps=$num_data_reps
    else
      num_reps=1
    fi
    python steps/data/reverberate_data_dir.py \
      --prefix "rev" \
      --rir-list-file data/impulses_noises/info/rir_list \
      --noise-list-file data/impulses_noises/info/noise_list \
      --foreground-snrs $foreground_snrs \
      --background-snrs $background_snrs \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications $num_reps \
      --max-noises-per-minute 1 \
      --random-seed 1 \
      data/${data_dir} data/${data_dir}_rvb
  done

  # copy the alignments for the newly created utterance ids
  ali_dirs=
  for i in `seq 1 $num_data_reps`; do
    local/multi_condition/copy_ali_dir.sh --utt-prefix "rev${i}_" exp/tri5a exp/tri5a_temp_$i || exit 1;
    ali_dirs+=" exp/tri5a_temp_$i"
  done
  steps/combine_ali_dirs.sh data/train_rvb exp/tri5a_rvb_ali $ali_dirs || exit 1;

  # copy the alignments for training the 100k system (from tri4a)
  local/multi_condition/copy_ali_dir.sh --utt-prefix "rev1_" exp/tri4a exp/tri4a_rvb || exit 1;
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc_reverb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_reverb-$date/s5/$mfccdir/storage $mfccdir/storage
  fi
  for data_dir in train_rvb dev_rvb test_rvb dev_aspire dev test ; do
    utils/copy_data_dir.sh data/$data_dir data/${data_dir}_hires
    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${data_dir}_hires \
        exp/make_reverb_hires/${data_dir} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${data_dir}_hires exp/make_reverb_hires/${data_dir} $mfccdir || exit 1;
    utils/fix_data_dir.sh data/${data_dir}_hires
    utils/validate_data_dir.sh data/${data_dir}_hires
  done

  # want the 100k subset to exactly match train_100k, since we'll use its alignments.
  awk -v p='rev1_' '{printf "%s%s\n", p, $1}' data/train_100k/utt2spk > uttlist
  #while read line; do grep $line data/train_rvb_hires/utt2spk|head -1; done < uttlist |awk '{print $1}' > uttlist2
  #mv uttlist2 uttlist
  utils/subset_data_dir.sh --utt-list uttlist \
    data/train_rvb_hires data/train_rvb_hires_100k
  rm uttlist
  utils/subset_data_dir.sh data/train_rvb_hires 30000 data/train_rvb_hires_30k
fi

if [ $stage -le 3 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/train_rvb_hires_100k data/lang exp/tri4a_rvb exp/nnet2_multicondition/tri5a
fi


if [ $stage -le 4 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest
  # subset.  the input directory exp/nnet2_online/tri5a is only needed for
  # the splice-opts and the LDA transform.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/train_rvb_hires_30k 512 exp/nnet2_multicondition/tri5a \
    exp/nnet2_multicondition/diag_ubm
fi

if [ $stage -le 5 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 100k subset (about one sixteenth of the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_rvb_hires_100k exp/nnet2_multicondition/diag_ubm \
    exp/nnet2_multicondition/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  ivectordir=exp/nnet2_multicondition/ivectors_train
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_reverb/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/train_rvb_hires data/train_rvb_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/train_rvb_hires_max2 exp/nnet2_multicondition/extractor $ivectordir || exit 1;
fi
