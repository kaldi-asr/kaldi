
set -u
set -e 
set -o pipefail

. path.sh
. cmd.sh

stage=-2
file_nj=40
nj=100
vad_dir=
ali_dir=
graph_dir=exp/tri4a/graph
transform_dir=
map_noise_to_sil=true

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <data-dir> <corrupted-data-dir> <lang> <model-dir> <dir>"
  echo " e.g.: $0 data/train_100k data/train_100k_corrupted exp/tri4a_ali_100k exp/vad_data_prep"
  exit 1
fi

data_dir=$1
corrupted_data_dir=$2
lang=$3
model_dir=$4
dir=$5

mkdir -p $dir

data_id=$(basename $data_dir)

utils/split_data.sh $data_dir $file_nj

if [ -z "$ali_dir" ]; then
  ali_dir=$dir/`basename ${model_dir}`_ali_${data_id}
  if [ $stage -le -2 ]; then
    steps/align_si.sh --cmd "$train_cmd" --nj $nj \
      $data_dir $lang $model_dir $ali_dir || exit 1
  fi
fi

if [ ! -f conf/phone_map_vad ]; then
  echo "$0: Expecting conf/phone_map_vad to exist!" && exit 1
fi

phone_map=conf/phone_map_vad 

if $map_noise_to_sil; then
  cat $phone_map | awk '{if ($2 == 2) print $1" 0"; else print $0}' > $dir/phone_map
  phone_map=$dir/phone_map
fi

if [ -z "$vad_dir" ]; then
  vad_dir=$dir/`basename ${model_dir}`_vad_${data_id}
  if [ $stage -le -1 ]; then
    diarization/convert_ali_to_vad.sh --phone-map $phone_map \
      --cmd "$train_cmd" \
      $data_dir $lang $ali_dir $vad_dir || exit 1
  fi
fi

if [ $stage -le 0 ]; then
  $train_cmd JOB=1:$file_nj $dir/log/get_file_lengths.JOB.log \
    wav-to-duration scp:$data_dir/split$file_nj/JOB/wav.scp \
    ark,t:- \| awk \'\{print \$1 " " int\(\$2 \* 100\)\}\' '>' $dir/file_lengths.JOB.ark || exit 1
fi

extended_data_dir=$dir/${data_id}_extended
if [ $stage -le 1 ]; then
  rm -rf $extended_data_dir
  mkdir -p $extended_data_dir/split$file_nj
  utils/copy_data_dir.sh $data_dir $extended_data_dir
  for f in cmvn.scp feats.scp text; do
    rm -f $extended_data_dir/$f
  done

  $train_cmd JOB=1:$file_nj $dir/log/get_empty_segments.JOB.log \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=1 --ignore-missing=false \
    "ark:segmentation-init-from-lengths --label=0 ark:$dir/file_lengths.JOB.ark ark:- |" \
    "ark:segmentation-init-from-segments $data_dir/split$file_nj/JOB/segments ark:- |" \
    ark:- \| segmentation-post-process --remove-labels=1 ark:- ark:- \| \
    segmentation-post-process --max-segment-length=1000 --post-process-label=0 \
    ark:- ark:- \| segmentation-to-segments --single-speaker=true --frame-overlap=0 \
    ark:- ark,t:$extended_data_dir/split$file_nj/utt2spk_empty.JOB \
    ark,t:$extended_data_dir/split$file_nj/segments_empty.JOB || exit 1
fi

if [ $stage -le 2 ] ; then
  for n in `seq $file_nj`; do 
    cat $extended_data_dir/split$file_nj/utt2spk_empty.$n $data_dir/split$file_nj/$n/utt2spk | sort -k1,1 | tee $extended_data_dir/split$file_nj/utt2spk.$n
  done > $extended_data_dir/utt2spk

  [ ! -s $extended_data_dir/utt2spk ] && echo "$0: $extended_data_dir/utt2spk is empty!" && exit 1

  for n in `seq $file_nj`; do
    cat $extended_data_dir/split$file_nj/segments_empty.$n $data_dir/split$file_nj/$n/segments | sort -k1,1 | tee $extended_data_dir/split$file_nj/segments.$n
  done > $extended_data_dir/segments

  utils/utt2spk_to_spk2utt.pl $extended_data_dir/utt2spk > $extended_data_dir/spk2utt
  utils/fix_data_dir.sh $extended_data_dir
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir/split$file_nj
  for n in `seq $file_nj`; do
    cat $extended_data_dir/split$file_nj/utt2spk_empty.$n | awk '{print $1}' > \
      $extended_data_dir/split$file_nj/text_empty.$n || exit 1
    cat $data_dir/split$file_nj/$n/text $extended_data_dir/split$file_nj/text_empty.$n | sort -k1,1 || tee $extended_data_dir/split$file_nj/text.$n
  done > $extended_data_dir/text
  utils/fix_data_dir.sh $extended_data_dir
fi

[ ! -s $vad_dir/vad.scp ] && echo "$0: $vad_dir/vad.scp is empty" && exit 1
if [ $stage -le 4 ]; then
  mkdir -p $dir/vad
  for n in `seq $file_nj`; do
    utils/filter_scp.pl $data_dir/split$file_nj/$n/utt2spk $vad_dir/vad.scp > \
      $dir/vad/vad_tmp.$n.scp || exit 1
    [ ! -s $dir/vad/vad_tmp.$n.scp ] && echo "$0: no utterances in $dir/vad/vad_tmp.$n.scp" && exit 1
  done
  
  $train_cmd JOB=1:$file_nj $dir/log/get_empty_vad.JOB.log \
    segmentation-init-from-segments --label=0 --per-utt=true $extended_data_dir/split$file_nj/segments_empty.JOB ark:- \| \
    segmentation-to-ali ark:- ark,scp:$dir/vad/vad_empty.JOB.ark,$dir/vad/vad_empty.JOB.scp

  for n in `seq $file_nj`; do
    cat $dir/vad/vad_tmp.$n.scp $dir/vad/vad_empty.$n.scp | sort -k 1,1 | tee $dir/vad/vad.$n.scp
  done > $dir/vad/vad.scp
fi

if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf ${extended_data_dir} \
    exp/make_mfcc/${data_id}_whole mfcc || exit 1
  cp $data_dir/cmvn.scp $extended_data_dir
  utils/fix_data_dir.sh $extended_data_dir
fi

[ -z "$model_dir" ] && model_dir=$ali_dir
[ -z "$graph_dir" ] && graph_dir=$model_dir/graph

if [ $stage -le 7 ]; then
  if [ ! -d $graph_dir ]; then
    utils/mkgraph.sh ${lang}_test $model_dir $graph_dir || exit 1
  fi
fi

if [ $stage -le 8 ]; then
  steps/decode_nolats.sh --cmd "$train_cmd --mem 2G" --nj $nj --transform-dir "$transform_dir" \
    --max-active 1000 --beam 10.0 --write-words false --write-alignments true \
    $graph_dir ${extended_data_dir} ${model_dir}/decode_${data_id}_whole || exit 1
fi

decode_vad_dir=$dir/${model_dir}_decode_vad_${data_id}
if [ $stage -le 9 ]; then
  diarization/convert_ali_to_vad.sh --phone-map $phone_map \
    --cmd "$train_cmd" --model $model_dir/final.mdl \
    $extended_data_dir $graph_dir $model_dir/decode_${data_id}_whole $decode_vad_dir || exit 1
fi

if [ $stage -le 10 ]; then
  vad_scps=()
  mkdir -p $dir/vad/split$nj
  mkdir -p $decode_vad_dir/split$nj
  for n in `seq $nj`; do
    vad_scps+=($dir/vad/split$nj/vad.$n.scp)
  done
  utils/split_scp.pl $dir/vad/vad.scp ${vad_scps[@]}
  
  mkdir -p $dir/intersected_segmentations
  $train_cmd JOB=1:$nj $dir/log/intersect_segments_empty.JOB.log \
    utils/filter_scp.pl $data_dir/utt2spk $dir/vad/split$nj/vad.JOB.scp \
    '>' $dir/vad/split$nj/vad_tmp.JOB.scp '&&' \
    utils/filter_scp.pl --exclude $data_dir/utt2spk $dir/vad/split$nj/vad.JOB.scp \
    '>' $dir/vad/split$nj/vad_empty.JOB.scp '&&' \
    utils/filter_scp.pl $dir/vad/split$nj/vad_tmp.JOB.scp $decode_vad_dir/vad.scp \
    '>' $decode_vad_dir/split$nj/vad_tmp.JOB.scp '&&' \
    utils/filter_scp.pl $dir/vad/split$nj/vad_empty.JOB.scp $decode_vad_dir/vad.scp \
    '>' $decode_vad_dir/split$nj/vad_empty.JOB.scp '&&' \
    segmentation-intersect-segments --mismatch-label=10 \
    "ark:segmentation-init-from-ali scp:$dir/vad/split$nj/vad_empty.JOB.scp ark:- |" \
    "ark:segmentation-init-from-ali scp:$decode_vad_dir/split$nj/vad_empty.JOB.scp ark:- |"  \
    ark,scp:$dir/intersected_segmentations/intersected_segmentations_empty.JOB.ark,$dir/intersected_segmentations/intersected_segmentations_empty.JOB.scp '&&' \
    segmentation-init-from-ali scp:$dir/vad/split$nj/vad_tmp.JOB.scp \
    ark,scp:$dir/intersected_segmentations/intersected_segmentations_tmp.JOB.ark,$dir/intersected_segmentations/intersected_segmentations_tmp.JOB.scp || exit 1

  for n in `seq $nj`; do 
    cat $dir/intersected_segmentations/intersected_segmentations_empty.$n.scp 
    cat $dir/intersected_segmentations/intersected_segmentations_tmp.$n.scp 
  done > $dir/intersected_segmentations/final_segmentations.scp
fi

#if [ $stage -le 11 ]; then
#  for n in `seq 
#  utils/split_data.sh $extended_data_dir $nj
#  
#  $train_cmd JOB=1:$nj $dir/log/post_process_intersected_orig_segmentations.JOB.log \
#    utils/filter_scp.pl $data_dir/utt2spk $dir/intersected_segmentations.JOB.scp \| \
#    segmentation-post-process --remove-labels=10 --merge-adjacent-segments=true \
#    --max-intersegment-length=10 scp:- \
#    ark,scp:$dir/intersected_segmentations/intersected_segmentations_tmp.JOB.ark,$dir/intersected_segmentations/intersected_segmentations_tmp.JOB.scp || exit 1
#
#  $train_cmd JOB=1:$nj $dir/log/create_final_segmentations.JOB.log \
#    utils/filter_scp.pl --exclude $dir/intersected_segmentations/intersected_segmentations_tmp.JOB.scp \
#    $dir/intersected_segmentations/intersected_segmentations.JOB.scp \| \
#    cat $dir/intersected_segmentations/intersected_segmentations_tmp.JOB.scp - \| \
#    sort -k1,1 \| segmentation-post-process --remove-labels=10 --merge-adjacent-segments=true \
#    scp:- ark,scp:$dir/intersected_segmentations/final_segmentations.JOB.ark,$dir/intersected_segmentations/final_segmentations.JOB.scp || exit 1
#
#  for n in `seq $nj`; do 
#    cat $dir/intersected_segmentations/final_segmentations.$n.scp
#  done > $dir/intersected_segmentations/final_segmentations.scp
#fi

if [ $stage -le 12 ]; then
  awk '{print $1" "$2}' $extended_data_dir/segments | \
    utils/utt2spk_to_spk2utt.pl > $extended_data_dir/reco2utt

  mkdir -p $dir/file_vad

  reco2utts=()
  for n in `seq $file_nj`; do
    reco2utts+=($extended_data_dir/split$file_nj/reco2utt.$n)
  done
  utils/split_scp.pl $extended_data_dir/reco2utt ${reco2utts[@]}

  $train_cmd JOB=1:$file_nj $dir/log/get_file_vad.JOB.log \
    utils/spk2utt_to_utt2spk.pl $extended_data_dir/split$file_nj/reco2utt.JOB '>' $extended_data_dir/split$file_nj/utt2reco.JOB '&&' \
    segmentation-combine-segments \
    "scp:utils/filter_scp.pl $extended_data_dir/split$file_nj/utt2reco.JOB $dir/intersected_segmentations/final_segmentations.scp |" \
    "ark,t:utils/filter_scp.pl $extended_data_dir/split$file_nj/utt2reco.JOB $extended_data_dir/segments |" \
    ark,t:$extended_data_dir/split$file_nj/reco2utt.JOB ark:- \| \
    segmentation-post-process --remove-labels=3:4 --merge-adjacent-segments=true ark:- ark:- \| \
    segmentation-to-ali --default-label=4 --lengths="ark:cat $dir/file_lengths.*.ark |" \
    ark:- ark,scp:$dir/file_vad/vad.JOB.ark,$dir/file_vad/vad.JOB.scp || exit 1

  for n in `seq $file_nj`; do 
    cat $dir/file_vad/vad.$n.scp
  done > $dir/file_vad/vad.scp
fi

#vad_data_dir=$dir/${data_id}_vad
#if [ $stage -le 13 ]; then
#  diarization/convert_data_dir_to_whole.sh $extended_data_dir $vad_data_dir
#  utils/fix_data_dir.sh ${vad_data_dir}
#  
#  utils/copy_data_dir.sh ${vad_data_dir} ${vad_data_dir}_hires
#  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf ${vad_data_dir}_hires exp/make_hires/${data_id}_vad mfcc_hires
#  steps/compute_cmvn_stats.sh ${vad_data_dir}_hires exp/make_hires/${data_id}_vad mfcc_hires
#  utils/fix_data_dir.sh ${vad_data_dir}_hires
#  
#  utils/copy_data_dir.sh ${vad_data_dir} ${vad_data_dir}_fbank
#  steps/make_fbank.sh --fbank-config conf/fbank.conf ${vad_data_dir}_fbank exp/make_fbank/${data_id}_vad fbank
#  steps/compute_cmvn_stats.sh --fake ${vad_data_dir}_fbank exp/make_fbank/${data_id}_vad mfcc_fbank
#  utils/fix_data_dir.sh ${vad_data_dir}_fbank
#fi
 
if [ $stage -le 13 ]; then
  local/snr/prepare_vad_training_data.sh --nj $file_nj --cmd "$train_cmd" \
    --keep-only-speech false $dir/file_vad/ $dir/file_vad || exit 1
fi

#  $train_cmd JOB=1:$file_nj $dir/file_vad/split$file_nj/log/get_segments.JOB.log \
#    segmentation-init-from-ali ark:$dir/file_vad/vad.JOB.ark ark:- \| \
#    segmentation-post-process --remove-labels=4:10 --merge-labels=0:1 --merge-dst-label=1 \
#    --shrink-length=20 --shrink-label=1 --merge-adjacent-segments=true --max-intersegment-length=1 \
#    ark:- ark:- \| segmentation-to-segments --single-speaker=true ark:- \
#    ark,t:$dir/file_vad/split$file_nj/reco2utt.JOB ark,t:$dir/file_vad/split$file_nj/segments.JOB || exit 1
#
#  for n in `seq $file_nj`; do
#    cat $dir/file_vad/split$file_nj/reco2utt.$n
#  done > $dir/file_vad/reco2utt
#
#  for n in `seq $file_nj`; do
#    cat $dir/file_vad/split$file_nj/segments.$n
#  done > $dir/file_vad/segments
#fi

