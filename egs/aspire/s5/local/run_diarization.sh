#LDA_MLLT_transform=exp/nnet/final.mat
#nnet=exp/nnet/final.mdl
#mfccdir=`pwd`/mfcc
#vaddir=`pwd`/mfcc
#trials_female=data/sre10_test_female/trials
#trials_male=data/sre10_test_male/trials
#
#steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
#    data/callhome exp/make_mfcc $mfccdir
#utils/fix_data_dir.sh data/callhome
#
#sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#    data/callhome exp/make_vad $vaddir
#
#sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=10G,ram_free=10G" \
#   exp/extractor  data/callhome \
#   exp/callhome
#
#ivector-normalize-length scp:exp/callhome/ivector.scp ark:- | ivector-subtract-global-mean ark:- ark:ivec1.ark
#ivector-subtract-global-mean scp:exp/callhome/ivector.scp ark:ivec2.ark
#
## The version 2 assumes that there are a variable number of speakers. 
## it does a kind of greedy clustering. If you remove the _v2 and the threshold
## argument, it then does K-Means clustering and assumes that you have only 2 speakers.
#speaker-diarization_v2 --threshold=-100 plda ark:data/callhome/spk2utt \
#  ark:feat_len.ark \
#  ark:ivec2.ark ark,t:diar_results.txt

. cmd.sh
. path.sh
set -e
set -o pipefail 

overlap=0.5
window=1.5
silence_weight=0.00001
max_count=100 # parameter for extract_ivectors.sh
mfccdir=mfcc_diarization
stage=-1
nj=30

. utils/parse_options.sh

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options] <data-dir> <file-weights-rspecifier> <extractor> <plda> <dir>"
  echo " e.g.: $0 data/dev_aspire_whole \"ark:gunzip -c exp/nnet2_multicondition/ivector_weights_dev_aspire_whole/file_weights.ark.gz |\" exp/nnet2_multicondition/ivector_extractor exp/nnet2_multicondition/diarization_dev_aspire_whole"
  echo " Options:"
  echo "    --stage (0|1|2)                 # start script from part-way through."
  exit 1
fi

data_dir=$1
lang=$2
file_weights=$3
extractor=$4
plda=$5
dir=$6

echo "$0: file weights are ignored by this script"

data_id=`basename $data_dir`
segmented_data_dir=${data_dir}_uniformsegmented_win${window}_over${overlap}

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh --validate-opts "--no-text" $data_dir $segmented_data_dir || exit 1
  cp $data_dir/reco2file_and_channel $segmented_data_dir || exit 1

  local/multi_condition/create_uniform_segments.py --overlap $overlap --window $window $segmented_data_dir || exit 1
  for file in cmvn.scp feats.scp text; do 
    rm -f $segmented_data_dir/$file
  done
fi

utils/validate_data_dir.sh --no-text --no-feats $segmented_data_dir || exit 1

segmented_data_id=`basename $segmented_data_dir`

if [ $stage -le 1 ]; then 
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj \
    --cmd "$train_cmd" $segmented_data_dir \
    exp/make_mfcc_diarization/$segmented_data_id $mfccdir || exit 1
  steps/compute_cmvn_stats.sh $segmented_data_dir \
    exp/make_mfcc_diarization/$segmented_data_id $mfccdir || exit 1
  utils/fix_data_dir.sh $segmented_data_dir
  utils/validate_data_dir.sh --no-text $segmented_data_dir || exit 1
fi

#if [ $stage -le 2 ]; then 
#  $train_cmd $dir/ivector_weights/log/extract_weights.log \
#    extract-vector-segments --trim-last-frames=2 --max-overshoot=0.025 \
#    "$file_weights" $segmented_data_dir/segments \
#    "ark:| gzip -c > $dir/ivector_weights/weights.gz" || exit 1
#fi

if [ $stage -le 3 ]; then
  diarization/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
    --silence-weight $silence_weight --max-count $max_count \
    --ivector-period 1 \
    $segmented_data_dir $lang $extractor \
    $dir/ivectors || exit 1
    #$dir/ivector_weights/weights.gz $dir/ivectors || exit 1
fi

utils/split_data.sh $segmented_data_dir $nj || exit 1

if [ $stage -le 4 ]; then
  $train_cmd JOB=1:$nj $dir/diarization/log/compute_feat_len.JOB.log \
    feat-to-len scp:$segmented_data_dir/split$nj/JOB/feats.scp \
    ark,t:$dir/diarization/feat_len.JOB.txt || exit 1
fi

if [ $stage -le 5 ]; then
  #$train_cmd JOB=1:$nj $dir/diarization/log/do_diarization.JOB.log \
  #  speaker-diarization_v2 --threshold=-100 $plda \
  #  ark:$segmented_data_dir/split$nj/JOB/spk2utt \
  #  ark,t:$dir/diarization/feat_len.JOB.txt \
  #  "scp:utils/filter_scp.pl $segmented_data_dir/split$nj/JOB/utt2spk $dir/ivectors/ivectors_utt.scp |" \
  #  ark,t:$dir/diarization/diarization_results.JOB.txt || exit 1
  plda=
  $train_cmd JOB=1:$nj $dir/diarization/log/do_diarization.JOB.log \
    speaker-diarization --num-speakers=3 $plda \
    ark:$segmented_data_dir/split$nj/JOB/spk2utt \
    "scp:utils/filter_scp.pl $segmented_data_dir/split$nj/JOB/utt2spk $dir/ivectors/ivectors_utt.scp |" \
    ark,t:$dir/diarization/diarization_results.JOB.txt || exit 1
fi

if [ $stage -le 6 ]; then
  mkdir -p $dir/diarization/data_out
  $train_cmd JOB=1:$nj $dir/diarization/log/convert_diarization_to_segmentation.JOB.log \
    segmentation-init-from-diarization --diarization-window-overlap=0.5 \
    ark,t:$dir/diarization/diarization_results.JOB.txt \
    $segmented_data_dir/split$nj/JOB/segments \
    ark,scp:$dir/diarization/diarization_segmentation.JOB.ark,$dir/diarization/diarization_segmentation.JOB.scp  || exit 1
  $train_cmd JOB=1:$nj $dir/diarization/log/convert_diarization_segmentation_to_segments.JOB.log \
    segmentation-to-segments ark:$dir/diarization/diarization_segmentation.JOB.ark ark,t:$dir/diarization/data_out/utt2spk.JOB \
    $dir/diarization/data_out/segments.JOB || exit 1
fi


