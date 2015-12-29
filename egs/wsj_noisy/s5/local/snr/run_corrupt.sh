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
corruption_stage=-10
pad_silence=false
mfcc_config=conf/mfcc_hires.conf
fbank_config=conf/fbank.conf
data_only=true

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

if [ $stage -le $num_data_reps ]; then
  corrupted_data_dirs=
  start_state=1
  if [ $stage -gt 1 ]; then 
    start_stage=$stage
  fi
  for x in `seq $start_stage $num_data_reps`; do
    cur_dest_dir=data/temp_`basename ${data_dir}`_$x
    output_clean_dir=data/temp_clean_`basename ${data_dir}`_$x
    output_noise_dir=data/temp_noise_`basename ${data_dir}`_$x
    local/snr/corrupt_data_dir.sh --random-seed $x --dest-wav-dir $dest_wav_dir/corrupted$x \
      --output-clean-wav-dir $dest_wav_dir/clean$x --output-clean-dir $output_clean_dir \
      --output-noise-wav-dir $dest_wav_dir/noise$x --output-noise-dir $output_noise_dir \
      --pad-silence $pad_silence --stage $corruption_stage --tmp-dir exp/make_corrupt/$x \
      --nj $nj $data_dir data/impulse_noises $cur_dest_dir
    corrupted_data_dirs+=" $cur_dest_dir"
    clean_data_dirs+=" $output_clean_dir"
    noise_data_dirs+=" $output_noise_dir"
  done

  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_corrupted ${corrupted_data_dirs}
  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_clean ${clean_data_dirs}
  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_noise ${noise_data_dirs}
  rm -rf $corrupted_data_dirs
  rm -rf $clean_data_dirs
fi

data_id=`basename $data_dir`
corrupted_data_dir=${data_dir}_corrupted
corrupted_data_id=`basename $corrupted_data_dir`
clean_data_dir=${data_dir}_clean
clean_data_id=`basename $clean_data_dir`
noise_data_dir=${data_dir}_noise
noise_data_id=`basename $noise_data_dir`

mfccdir=mfcc_hires
#if [ $stage -le 2 ]; then
#  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
#    date=$(date +'%m_%d_%H_%M')
#    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
#  fi
#
#  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_hires
#  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config $mfcc_config ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
#  steps/compute_cmvn_stats.sh ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
#  utils/fix_data_dir.sh ${clean_data_dir}_hires
#fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  utils/copy_data_dir.sh --extra-files utt2uniq ${corrupted_data_dir} ${corrupted_data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config $mfcc_config ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  steps/compute_cmvn_stats.sh ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_hires
fi

fbankdir=fbank_feats
if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi

  utils/copy_data_dir.sh --extra-files utt2uniq ${clean_data_dir} ${clean_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run 20" --nj $nj --fbank-config $fbank_config ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats
  steps/compute_cmvn_stats.sh --fake ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats
  utils/fix_data_dir.sh ${clean_data_dir}_fbank
fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi

  utils/copy_data_dir.sh --extra-files utt2uniq ${noise_data_dir} ${noise_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run 20" --nj $nj --fbank-config $fbank_config ${noise_data_dir}_fbank exp/make_fbank/${noise_data_id} fbank_feats
  steps/compute_cmvn_stats.sh --fake ${noise_data_dir}_fbank exp/make_fbank/${noise_data_id} fbank_feats
  utils/fix_data_dir.sh ${noise_data_dir}_fbank
fi

if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi
  utils/copy_data_dir.sh --extra-files utt2uniq ${corrupted_data_dir} ${corrupted_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run 20" --nj $nj --fbank-config $fbank_config ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats
  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_fbank
fi

[ $(cat ${clean_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${clean_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1

[ $(cat ${noise_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${noise_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1

$data_only && echo "--data-only is true" && exit 1

tmpdir=exp/make_irm_targets
targets_dir=irm_targets
if [ $stage -le 16 ]; then
  utils/split_data.sh ${clean_data_dir}_fbank $nj
  utils/split_data.sh ${noise_data_dir}_fbank $nj

  sleep 2

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$targets_dir/storage $targets_dir/storage
    for n in `seq $nj`; do 
      utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
    done
  fi

  mkdir -p $targets_dir 
  $train_cmd --max-jobs-run 30 JOB=1:$nj $tmpdir/${tmpdir}_${data_id}.JOB.log \
    compute-snr-targets --target-type="Irm" \
    scp:${clean_data_dir}_fbank/split$nj/JOB/feats.scp \
    scp:${noise_data_dir}_fbank/split$nj/JOB/feats.scp \
    ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $targets_dir/${data_id}.$n.scp
  done > ${corrupted_data_dir}_hires/`basename $targets_dir`.scp
fi

exit 0

tmpdir=exp/make_fbank_mask_targets
targets_dir=fbank_mask_targets
if [ $stage -le 17 ]; then
  utils/split_data.sh ${corrupted_data_dir}_fbank $nj
  utils/split_data.sh ${clean_data_dir}_fbank $nj

  sleep 2

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$targets_dir/storage $targets_dir/storage
    for n in `seq $nj`; do 
      utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
    done
  fi

  mkdir -p $targets_dir 
  $train_cmd --max-jobs-run 30 JOB=1:$nj $tmpdir/${tmpdir}_${data_id}.JOB.log \
    compute-snr-targets --target-type="FbankMask" \
    scp:${clean_data_dir}_fbank/split$nj/JOB/feats.scp \
    scp:${corrupted_data_dir}_fbank/split$nj/JOB/feats.scp \
    ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $targets_dir/${data_id}.$n.scp
  done > ${corrupted_data_dir}_hires/`basename $targets_dir`.scp
fi

tmpdir=exp/make_snr_targets
targets_dir=snr_targets
if [ $stage -le 18 ]; then
  utils/split_data.sh ${clean_data_dir}_fbank $nj
  utils/split_data.sh ${noise_data_dir}_fbank $nj
  
  sleep 2

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$targets_dir/storage $targets_dir/storage
    for n in `seq $nj`; do 
      utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
    done
  fi

  mkdir -p $targets_dir 
  $train_cmd --max-jobs-run 30 JOB=1:$nj $tmpdir/${tmpdir}_${data_id}.JOB.log \
    compute-snr-targets --target-type="Snr" \
    scp:${clean_data_dir}_fbank/split$nj/JOB/feats.scp \
    scp:${noise_data_dir}_fbank/split$nj/JOB/feats.scp \
    ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $targets_dir/${data_id}.$n.scp
  done > ${corrupted_data_dir}_hires/`basename $targets_dir`.scp
fi

tmpdir=exp/make_frame_snr_correct_targets
targets_dir=frame_snr_correct_targets
if [ $stage -le 19 ]; then
  utils/split_data.sh ${clean_data_dir}_fbank $nj
  utils/split_data.sh ${noise_data_dir}_fbank $nj

  sleep 2
  
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$targets_dir/storage $targets_dir/storage
    for n in `seq $nj`; do 
      utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
    done
  fi

  mkdir -p $targets_dir 
  $train_cmd JOB=1:$nj $tmpdir/${tmpdir}_${data_id}.JOB.log \
    matrix-sum --scale1=1.0 --scale2=-1.0 \
    "ark:compute-mfcc-feats --config=conf/mfcc.conf --num-ceps=1 --num-mel-bins=3 scp:${clean_data_dir}_fbank/split$nj/JOB/wav.scp ark:- |" \
    "ark:compute-mfcc-feats --config=conf/mfcc.conf --num-ceps=1 --num-mel-bins=3 scp:${noise_data_dir}_fbank/split$nj/JOB/wav.scp ark:- |" \
    ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $targets_dir/${data_id}.$n.scp
  done > ${corrupted_data_dir}_hires/`basename $targets_dir`.scp
fi

tmpdir=exp/make_frame_snr_targets
targets_dir=frame_snr_targets
if [ $stage -le 20 ]; then
  utils/split_data.sh ${clean_data_dir}_fbank $nj
  utils/split_data.sh ${noise_data_dir}_fbank $nj

  sleep 2

  mkdir -p $targets_dir 
  $train_cmd --max-jobs-run 30 JOB=1:$nj $tmpdir/${tmpdir}_${data_id}.JOB.log \
    vector-sum \
    "ark:matrix-scale scp:${clean_data_dir}_fbank/split$nj/JOB/feats.scp ark:- | matrix-sum-cols --log-sum-exp=true ark:- ark:- |" \
    "ark:matrix-scale scp:${noise_data_dir}_fbank/split$nj/JOB/feats.scp ark:- | matrix-sum-cols --log-sum-exp=true ark:- ark:- | vector-scale --scale=-1.0 ark:- ark:- |" \
    ark:- \| vector-to-feat ark:- \
    ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $targets_dir/${data_id}.$n.scp
  done > ${corrupted_data_dir}_hires/`basename $targets_dir`.scp
fi
