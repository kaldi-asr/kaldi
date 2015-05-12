set -o pipefail

. path.sh

cmd=run.pl
nj=4
stage=-1
get_whole_vad=false

. parse_options.sh

if [ $# -gt 4 ]; then
  echo "Usage: convert_ali_to_vad.sh <data-dir> <lang-dir> <ali-dir> [<vad-dir>]"
  echo " e.g.: convert_ali_to_vad.sh <data-dir> data/lang exp/tri5_ali"
  exit 1
fi

data=$1
lang=$2
ali_dir=$3
if [ $# -eq 4 ]; then
  dir=$4
else 
  dir=$ali_dir/vad
fi

for f in $lang/phones/silence.int $lang/phones/nonsilence.int $ali_dir/ali.1.gz $ali_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: $f does not exist" && exit 1
done 

tmpdir=$dir/tmp
mkdir -p $tmpdir

num_jobs=`cat $ali_dir/num_jobs` || exit 1
echo $num_jobs > $dir/num_jobs

split_data.sh $data $num_jobs

{
awk '{print $1" 0"}' $lang/phones/silence.int;
awk '{print $1" 1"}' $lang/phones/nonsilence.int;
} > $dir/phone_map

if [ $stage -le 0 ]; then
  $cmd JOB=1:$num_jobs $dir/log/convert_ali_to_vad.JOB.log \
    ali-to-phones --per-frame=true $ali_dir/final.mdl "ark:gunzip -c $ali_dir/ali.JOB.gz |" \
    ark,t:- \| utils/apply_map.pl -f 2- $dir/phone_map \| \
    copy-int-vector ark,t:- "ark:| gzip -c > $tmpdir/vad.JOB.ark.gz" || exit 1
fi

if ! $get_whole_vad; then
  if [ $stage -le 1 ]; then
    $cmd JOB=1:$num_jobs $dir/log/combine_vad.JOB.log \
      copy-int-vector "ark:gunzip -c $tmpdir/vad.JOB.ark.gz |" \
      ark,scp:$dir/vad.JOB.ark,$dir/vad.JOB.scp || exit 1

    for n in `seq $num_jobs`; do 
      cat $dir/vad.$n.scp
    done | sort -k1,1 > $dir/vad.scp
  fi
else
  if [ $stage -le 1 ]; then
    $cmd JOB=1:$num_jobs $dir/log/get_whole_vad.JOB.log \
      copy-int-vector "ark:gunzip -c $tmpdir/vad.JOB.ark.gz |" ark,t:- \| \
      diarization/convert_vad_to_rttm.pl --silence-class 0 --speech-class 1 \
      --segments $data/split$num_jobs/JOB/segments \| rttmSort.pl \| \
      diarization/convert_rttm_to_vad.pl --ignore-boundaries true \
      --segments-out $tmpdir/segments.JOB \| \
      copy-int-vector ark,t:- ark,scp:$dir/vad.JOB.ark.gz,$dir/vad.JOB.scp || exit 1

    for n in `seq $num_jobs`; do 
      cat $dir/vad.$n.scp
    done | sort -k1,1 > $dir/vad.scp
  fi

  if [ $stage -le 2 ]; then
    rm -rf $dir/data
    mkdir -p $dir/data

    utils/copy_data_dir.sh $data $dir/data
    for f in feats.scp segments spk2utt utt2spk vad.scp cmvn.scp spk2gender text; do 
      [ -f $dir/data/$f ] && rm $dir/data/$f
    done
    rm -rf $dir/data/split* $dir/data/.backup

    eval cat $tmpdir/segments.{`seq -s',' $num_jobs`} | sort -k1,1 > $dir/data/segments

    awk '{print $1" "$2}' $dir/data/segments > $dir/data/utt2spk || exit 1
    utils/utt2spk_to_spk2utt.pl $dir/data/utt2spk > $dir/data/spk2utt || exit 1

    utils/fix_data_dir.sh $dir/data || exit 1

    rm -rf $dir/data_whole

    mkdir -p $dir/data_whole 
    utils/copy_data_dir.sh $data $dir/data_whole
    for f in feats.scp segments spk2utt utt2spk vad.scp cmvn.scp spk2gender text; do 
      [ -f $dir/data_whole/$f ] && rm $dir/data_whole/$f
    done
    rm -rf $dir/data_whole/split* $dir/data_whole/.backup

    awk '{print $1" "$1}' $dir/data_whole/wav.scp > $dir/data_whole/utt2spk || exit 1
    utils/utt2spk_to_spk2utt.pl $dir/data_whole/utt2spk > $dir/data_whole/spk2utt || exit 1

    utils/fix_data_dir.sh $dir/data_whole || exit 1
  fi

  if [ $stage -le 3 ]; then
    diarization/prepare_data.sh --cmd $cmd --nj $nj $dir/data_whole $dir/tmpdir $dir/mfcc || exit 1
  fi

  if [ $stage -le 4 ]; then
    split_data.sh $dir/data $nj
    $cmd JOB=1:$nj $dir/log/extract_feats.JOB.log \
      utils/filter_scp.pl $dir/data/split$nj/JOB/wav.scp $dir/data_whole/feats.scp \| \
      extract-feature-segments scp:- \
      $dir/data/split$nj/JOB/segments \
      ark,scp:$dir/mfcc/raw_mfcc_feats_data.JOB.ark,$dir/mfcc/raw_mfcc_feats_data.JOB.scp || exit 1

    eval cat $dir/mfcc/raw_mfcc_feats_data.{`seq -s',' $nj`}.scp | sort -k1,1 > $dir/data/feats.scp
  fi
fi
