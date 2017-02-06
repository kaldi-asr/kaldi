#! /bin/bash

window=2.5
overlap=0.0
stage=-1
cmd=queue.pl
reco_nj=4
frame_shift=0.01
utt_nj=18
min_clusters=10
clustering_opts="--stopping-threshold=0.5 --max-merge-thresh=0.25 --normalize-by-entropy"

. path.sh
. utils/parse_options.sh

set -o pipefail
set -e
set -u

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data> <dir> <out-data>"
  exit 1
fi

data=$1
dir=$2
out_data=$3

num_frames=`perl -e "print int($window / $frame_shift + 0.5)"`
num_frames_overlap=`perl -e "print int($overlap/ $frame_shift + 0.5)"`

data_uniform_seg=$dir/`basename ${data}`_uniform_seg_window${window}_ovlp${overlap}

mkdir -p ${data_uniform_seg}

mkdir -p $dir

#segmentation-cluster-adjacent-segments --verbose=0 'ark:segmentation-copy --keep-label=1 "ark:gunzip -c exp/nnet3_lstm_sad_music/nnet_lstm_1e//segmentation_bn_eval97_whole_bp/orig_segmentation.1.gz |" ark:- | segmentation-split-segments  --max-segment-length=250 --overlap-length=0 ark:- ark:- |' scp:data/bn_eval97_bp_hires/feats.scp "ark:| segmentation-post-process --merge-adjacent-segments ark:- ark:- | segmentation-to-segments ark:- ark,t:- /dev/null" 2>&1 | less

if [ $stage -le 0 ]; then
  $cmd $dir/log/get_subsegments.log \
    segmentation-init-from-segments --frame-overlap=0.015 $data/segments ark:- \| \
    segmentation-split-segments --max-segment-length=$num_frames --overlap-length=$num_frames_overlap ark:- ark:- \| \
    segmentation-to-segments --frame-overlap=0.0 ark:- ark:/dev/null \
    ${data_uniform_seg}/sub_segments

  utils/data/subsegment_data_dir.sh ${data} ${data_uniform_seg}{/sub_segments,}
fi

gmm_dir=$dir/gmms
mkdir -p $gmm_dir

utils/split_data.sh --per-reco ${data_uniform_seg} $reco_nj

if [ $stage -le 1 ]; then
  echo $reco_nj > $gmm_dir/num_jobs
  $cmd JOB=1:$reco_nj $gmm_dir/log/train_gmm.JOB.log \
    gmm-global-init-models-from-feats --share-covars=true \
    --spk2utt-rspecifier=ark,t:${data_uniform_seg}/split${reco_nj}reco/JOB/reco2utt \
    --num-gauss-init=64 --num-gauss=64 --num-gauss-fraction=0.001 --max-gauss=512 --min-gauss=64 \
    --num-iters=20 --num-frames=500000 \
    scp:${data_uniform_seg}/split${reco_nj}reco/JOB/feats.scp \
    ark,scp:$gmm_dir/gmm.JOB.ark,$gmm_dir/gmm.JOB.scp
  
  for n in `seq $reco_nj`; do
    cat $gmm_dir/gmm.$n.scp
  done > $gmm_dir/gmm.scp

fi

post_dir=$gmm_dir/post_`basename $data_uniform_seg`
mkdir -p $post_dir

if [ $stage -le 2 ]; then
  echo $reco_nj > $post_dir/num_jobs

  $cmd JOB=1:$reco_nj $gmm_dir/log/compute_post.JOB.log \
    gmm-global-get-post \
    --utt2spk="ark,t:cut -d ' ' -f 1,2 ${data_uniform_seg}/split${reco_nj}reco/JOB/segments |" \
    scp:$gmm_dir/gmm.scp \
    scp:${data_uniform_seg}/split${reco_nj}reco/JOB/feats.scp \
    "ark:| gzip -c > $post_dir/post.JOB.gz" \
    "ark:| gzip -c > $post_dir/frame_loglikes.JOB.gz"
fi

if [ $stage -le 3 ]; then
  utils/data/get_utt2num_frames.sh --nj $utt_nj --cmd "$cmd" ${data_uniform_seg}
  
  $cmd JOB=1:$reco_nj $post_dir/log/compute_average_post.JOB.log \
    gmm-global-post-to-feats \
    --utt2spk="ark,t:cut -d ' ' -f 1,2 ${data_uniform_seg}/split${reco_nj}reco/JOB/segments |" \
    scp:$gmm_dir/gmm.scp "ark:gunzip -c $post_dir/post.JOB.gz |" ark:- \| \
    matrix-sum-rows --do-average ark:- "ark:| gzip -c > $post_dir/avg_post.JOB.gz"
fi

seg_dir=$dir/segmentation_`basename $data_uniform_seg`

if [ $stage -le 4 ]; then
  $cmd JOB=1:$reco_nj $seg_dir/log/compute_scores.JOB.log \
    ib-scoring-dense --input-factor=0.0 $clustering_opts \
    --counts-rspecifier="ark,t:utils/filter_scp.pl $data_uniform_seg/split${reco_nj}reco/JOB/utt2spk $data_uniform_seg/utt2num_frames |" \
    "ark,t:${data_uniform_seg}/split${reco_nj}reco/JOB/reco2utt" \
    "ark:gunzip -c $post_dir/avg_post.JOB.gz |" \
    ark,t:$seg_dir/scores.JOB.txt ark:/dev/null
fi

if [ $stage -le 5 ]; then
  threshold=$(for n in `seq $reco_nj`; do
    /export/a12/vmanoha1/kaldi-diarization-v2/src/ivectorbin/compute-calibration \
      ark,t:$seg_dir/scores.$n.txt -; done | \
      awk '{i += $1; j++;} END{print i / j}')
  echo $threshold > $seg_dir/threshold 
fi

threshold=$(cat $seg_dir/threshold)
if [ $stage -le 6 ]; then
  $cmd JOB=1:$reco_nj $seg_dir/log/cluster_segments.JOB.log \
    agglomerative-cluster-ib --input-factor=0.0 --min-clusters=$min_clusters $clustering_opts \
    --max-merge-thresh=$threshold --verbose=3 \
    --counts-rspecifier="ark,t:utils/filter_scp.pl $data_uniform_seg/split${reco_nj}reco/JOB/utt2spk $data_uniform_seg/utt2num_frames |" \
    "ark:gunzip -c $post_dir/avg_post.JOB.gz |" \
    "ark,t:${data_uniform_seg}/split${reco_nj}reco/JOB/reco2utt" \
    ark,t:$seg_dir/utt2cluster_id.JOB
fi

if [ $stage -le 7 ]; then
  $cmd JOB=1:$reco_nj $seg_dir/log/init_segmentation.JOB.log \
    segmentation-init-from-segments --frame-overlap=0.0 --shift-to-zero=false \
    --utt2label-rspecifier=ark,t:${seg_dir}/utt2cluster_id.JOB \
    ${data_uniform_seg}/split${reco_nj}reco/JOB/segments ark:- \| \
    segmentation-combine-segments-to-recordings ark:- \
    ark,t:${data_uniform_seg}/split${reco_nj}reco/JOB/reco2utt \
    ark:- \| \
    segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
    segmentation-post-process --max-segment-length=1000 --overlap-length=250 ark:- ark:- \| \
    segmentation-to-segments ark:- ark,t:$seg_dir/utt2spk.JOB $seg_dir/segments.JOB
fi

if [ $stage -le 8 ]; then
  rm -r $out_data || true
  utils/data/convert_data_dir_to_whole.sh $data $out_data
  rm $out_data/{text,cmvn.scp} || true

  for n in `seq $reco_nj`; do
    cat $seg_dir/utt2spk.$n
  done > $out_data/utt2spk

  for n in `seq $reco_nj`; do
    cat $seg_dir/segments.$n
  done > $out_data/segments

  utils/utt2spk_to_spk2utt.pl $out_data/utt2spk > $out_data/spk2utt
  utils/fix_data_dir.sh $out_data
fi
