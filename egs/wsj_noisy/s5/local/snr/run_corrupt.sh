#!/bin/bash
set -e
set -o pipefail

. path.sh
. cmd.sh

num_data_reps=5
data_dir=data/train_si284
dest_wav_dir=wavs
nj=40
stage=1

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

if [ $stage -le 1 ]; then
  corrupted_data_dirs=
  for x in `seq $num_data_reps`; do
    cur_dest_dir=data/temp_`basename ${data_dir}`_$x
    output_clean_dir=data/temp_clean_`basename ${data_dir}`_$x
    local/snr/corrupt_data_dir.sh --random-seed $x --dest-wav-dir $dest_wav_dir/corrupted$x \
      --output-clean-wav-dir $dest_wav_dir/clean$x --output-clean-dir $output_clean_dir \
      $data_dir data/impulse_noises $cur_dest_dir
    corrupted_data_dirs+=" $cur_dest_dir"
    clean_data_dirs+=" $output_clean_dir"
  done

  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_corrupted ${corrupted_data_dirs}
  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_clean ${clean_data_dirs}
  rm -rf $corrupted_data_dirs
  rm -rf $clean_data_dirs
fi

data_id=`basename $data_dir`
corrupted_data_dir=${data_dir}_corrupted
corrupted_data_id=`basename $corrupted_data_dir`
clean_data_dir=${data_dir}_clean
clean_data_id=`basename $clean_data_dir`

mfccdir=mfcc_hires
if [ $stage -le 2 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
  steps/compute_cmvn_stats.sh ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
  utils/fix_data_dir.sh ${clean_data_dir}_hires
fi

if [ $stage -le 3 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  utils/copy_data_dir.sh --extra-files utt2uniq ${corrupted_data_dir} ${corrupted_data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  steps/compute_cmvn_stats.sh ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_hires
fi

fbankdir=fbank_feats
if [ $stage -le 4 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi

  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd" --nj $nj --fbank-config conf/fbank.conf ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats
  steps/compute_cmvn_stats.sh --fake ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats
  utils/fix_data_dir.sh ${clean_data_dir}_fbank
fi

if [ $stage -le 5 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi
  utils/copy_data_dir.sh --extra-files utt2uniq ${corrupted_data_dir} ${corrupted_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd" --nj $nj --fbank-config conf/fbank.conf ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats
  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_fbank
fi

tmpdir=exp/make_snr_targets

[ $(cat ${clean_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${clean_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1

if [ $stage -le 6 ]; then
  utils/split_data.sh ${corrupted_data_dir}_fbank $nj
  utils/split_data.sh ${clean_data_dir}_fbank $nj

  sleep 10

  mkdir -p snr_targets
  $train_cmd JOB=1:$nj exp/make_snr_targets/make_snr_targets_${data_id}.JOB.log \
    compute-snr-targets scp:${clean_data_dir}_fbank/split$nj/JOB/feats.scp \
    scp:${corrupted_data_dir}_fbank/split$nj/JOB/feats.scp \
    ark,scp:snr_targets/${data_id}.JOB.ark,snr_targets/${data_id}.JOB.scp

  for n in `seq $nj`; do
    cat snr_targets/${data_id}.$n.scp
  done > ${corrupted_data_dir}_hires/snr_targets.scp
fi
