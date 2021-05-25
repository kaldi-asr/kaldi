#!/bin/bash
# Copyright   2020   Ivan Medennikov

# Apache 2.0.
#
# This script performs 2nd and further iterations of TS-VAD diarization
# on a set of kinect channels followed by averaging.
# Probabilities from the previous iteration are used to estimate i-vectors

cmd="run.pl"
ref_rttm=
lang=data/lang
audio_dir=CHiME6/audio

#blstm processing parameters
extra_left_context=30
extra_right_context=30
frames_per_chunk=40

#post-processing parameters
thr=0.4
window=51
min_silence=0.3
min_speech=0.2

nj=8
nj_feats=2
piece=10000 #raw wavs will be splitted into non-overlapping pieces of this size (in frames)
ups=18
wpeid=
channels="CH1 CH2 CH3 CH4"

#parameters for modification of initial weights
t=0
mt=0.7 

it=2
ivector_affix=it1-init

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 4 ]; then
  echo "Usage: $0 <ts-vad-dir> <ivector-dir> <initname> <out-dir>"
  echo "e.g.: $0 exp/ts-vad exp/nnet3 exp/ts-vad/it1 exp/ts-vad/it2"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --ref_rttm ./chime6_rttm/dev_rttm                # the location of the reference RTTM file"
  echo "  --it 2                                           # current iteration of TS-VAD"
  echo "  --ivector_affix it1-init                         # affix corresponding to the initial weights"
  echo "  --channels CH1 CH2 CH3 CH4                       # kinect channels to be processed"
  echo "  --audio_dir CHiME6/audio                         # path to wav files"
  echo "  --wpeid WPE2m                                    # affix for non-original wavs, e.g., blockwise WPE processed"
  echo "  --piece 10000                                    # raw wavs will be splitted into non-overlapping pieces of this size (in frames)"
  echo "  --ups 18                                         # number of pieces considered as one speaker"
  echo "  --t 0                                            # absolute threshold for initial weights"
  echo "  --mt 0.7                                         # relative threshold for pi/(p1+p2+p3+p4) in initial weights (to exclude overlapping regions from i-vectors estimation)"
  echo "  --thr 0.4                                        # post-processing: probability threshold"
  echo "  --window 51                                      # post-processing: median filter window (in frames)"
  echo "  --min_silence 0.3                                # post-processing: minimum length of silence (in seconds)"
  echo "  --min_speech 0.2                                 # post-processing: minimum length of speech (in seconds)"
  exit 1;
fi

dir=$1
ivector_dir=$2
initdir=$3
outdir=$4

initname=$(basename $initdir)
test="$(cut -d'_' -f1 <<<"$initname")"

weights=$initdir/weights.ark
weights_mod=$initdir/weights_t${t}_mt${mt}.ark
if [ ! -f ${weights_mod}.gz ]; then
  python local/ts-vad/vad_prob_mod.py --threshold $t --multispk_threshold $mt ark:$weights ark,t:${weights_mod}
  cat ${weights_mod} | sed s/_U06.ENH// | sort | gzip -c > ${weights_mod}.gz
  rm $weights_mod
fi
for spk in `seq 4`; do
  [ ! -f ${weights_mod}.${spk}.gz ] && gunzip -c ${weights_mod}.gz | grep "\-$spk\ " | sed s/\-$spk\ /\ / | gzip -c > ${weights_mod}.${spk}.gz
done

kinects="U01 U02 U03 U04 U05 U06"
[ "$test" == "dev" ] && kinects="U01 U02 U03 U04 U06"
[ "$test" == "eval" ] && kinects="U01 U02 U04 U05 U06"
 
sum_scps=""
n=0
for u in $kinects; do
  for ch in $channels; do
    id=${u}.${ch}${wpeid}
    echo "processing $id"

    dset=${test}_${id}_hires
    if [ ! -f data/$dset/.done ]; then
      mkdir -p data/$dset
      ls $audio_dir/$test/ | grep "wav" | grep "$u" | grep "$ch" | awk -v "pth=$audio_dir/$test" '{printf "%s %s/%s\n", $1, pth, $1}' | sed -E s/_[^\ ]+// > data/$dset/wav.scp
      awk '{print $1" "$1}' data/$dset/wav.scp > data/$dset/utt2spk
      awk '{print $1" "$1}' data/$dset/wav.scp > data/$dset/spk2utt
      utils/fix_data_dir.sh data/$dset
      steps/make_mfcc.sh --nj $nj_feats --mfcc-config conf/mfcc_hires.conf data/$dset data/$dset/log data/$dset/data || exit 1;
      touch data/$dset/.done
    fi

    dsetsrc=$dset
    dset=${dset}_split${piece}
    if [ ! -f data/$dset/.done ]; then
      mkdir -p data/$dset
      cp data/${dsetsrc}/wav.scp data/$dset
      feat-to-len scp:data/$dsetsrc/feats.scp ark,t:data/$dsetsrc/utt2len
      local/ts-vad/split_feats_seg.pl data/$dsetsrc/feats.scp data/$dsetsrc/utt2spk data/$dsetsrc/utt2len $piece data/$dset/feats.scp data/$dset/utt2spk data/$dset/segments
      utils/utt2spk_to_spk2utt.pl data/$dset/utt2spk > data/$dset/spk2utt
      utils/fix_data_dir.sh data/$dset
      touch data/$dset/.done
    fi

    dsetsrc=$dset
    dset=${dset}_${ups}ups
    if [ ! -f data/$dset/.done ]; then
      utils/copy_data_dir.sh data/$dsetsrc data/$dset
      local/ts-vad/modify_ups_utt2spk.pl data/$dsetsrc/utt2spk $ups data/$dset/utt2spk
      utils/utt2spk_to_spk2utt.pl data/$dset/utt2spk > data/$dset/spk2utt
      utils/fix_data_dir.sh data/$dset
      touch data/$dset/.done
    fi

    ivdir=${ivector_dir}/${test}_${ivector_affix}/ivectors_${dset}
    for spk in `seq 4`; do
      if [ ! -f $ivdir/$spk/ivector_online.scp ]; then
        echo "Extracting i-vectors for $dset"
        steps/online/nnet2/extract_ivectors.sh --cmd "$decode_cmd" --nj $nj \
          --silence-weight 0.00001 \
          --sub-speaker-frames 0 --max-count 100 \
          data/$dset $lang $ivector_dir/extractor ${weights_mod}.${spk}.gz $ivdir/$spk || exit 1;
      fi
    done

    iv4dir=${ivector_dir}/${test}_${ivector_affix}/ivectors-4spk_${dset}
    if [ ! -f $iv4dir/.done ]; then
      mkdir -p $iv4dir
      echo "Making pseudo-online 4spk i-vectors using source $ivdir"
      for spk in `seq 4`; do
        cat $ivdir/$spk/ivectors_spk.*.ark > $iv4dir/ivectors_spk.$spk.ark
      done
      $train_cmd JOB=1:4 $iv4dir/log/apply-map.JOB.log \
        utils/apply_map.pl -f 2 $iv4dir/ivectors_spk.JOB.ark \<data/$dset/utt2spk \>$iv4dir/ivectors_utt.JOB.ark || exit 1;

      ivector_dim=$[$(head -n 1 $iv4dir/ivectors_spk.1.ark | wc -w) - 3] || exit 1;
      base_feat_dim=$(feat-to-dim scp:data/$dset/feats.scp -) || exit 1;
      start_dim=$base_feat_dim
      end_dim=$[$base_feat_dim+$ivector_dim-1]
      absdir=$(utils/make_absolute.sh $iv4dir)
      cp $ivdir/1/{ivector_period,final.ie.id} $iv4dir/
      ivector_period=$(cat $iv4dir/ivector_period)

      $train_cmd JOB=1:4 $iv4dir/log/duplicate_feats.JOB.log \
        append-vector-to-feats scp:data/$dset/feats.scp ark:$iv4dir/ivectors_utt.JOB.ark ark:- \| \
        select-feats "$start_dim-$end_dim" ark:- ark:- \| \
        subsample-feats --n=$ivector_period ark:- ark:- \| \
        copy-feats --compress=true ark:- \
        ark,scp:$absdir/ivector_online.JOB.ark,$absdir/ivector_online.JOB.scp || exit 1;

      $train_cmd $iv4dir/log/paste-feats.log \
        paste-feats scp:$iv4dir/ivector_online.1.scp scp:$iv4dir/ivector_online.2.scp scp:$iv4dir/ivector_online.3.scp scp:$iv4dir/ivector_online.4.scp ark:- \| \
        copy-feats --compress=true ark:- ark,scp:$absdir/ivector_online.ark,$absdir/ivector_online.scp || exit 1;
      touch $iv4dir/.done
    fi

    out=$outdir/$dset
    if [ ! -f $out/.done ]; then
      local/ts-vad/compute_ts-vad_weights.sh --nj $nj --use-gpu true --cmd "$decode_cmd" --online-ivector-dir $iv4dir \
        --extra-left-context $extra_left_context --extra-right-context $extra_right_context --frames-per-chunk $frames_per_chunk \
        data/$dset $dir/final.raw $out || exit 1;
      touch $out/.done
    fi
    sum_scps="${sum_scps}ark:$out/weights.ark "
    n=$((n+1))
  done
done

id=${n}ch-AVG${wpeid}
dset=${test}_${id}_hires_split${piece}_${ups}ups
out=$outdir/$dset
if [ ! -f $out/.done ]; then
  scale=$(awk -v "n=$n" 'BEGIN {print 1/n}')
  $train_cmd $out/log/vector-sum.log \
    vector-sum $sum_scps ark:- \| vector-scale --scale=$scale ark:- ark,t:$out/weights.ark || exit 1;
  touch $out/.done
fi

scoring=$out/scoring
hyp_rttm=$scoring/rttm
if [ ! -f $scoring/.done ]; then
  if [ ! -f $hyp_rttm ]; then 
    python local/ts-vad/convert_prob_to_rttm.py --threshold $thr --window $window --min_silence $min_silence --min_speech $min_speech ark:"sort $out/weights.ark |" $hyp_rttm || exit 1;
  fi
  echo "Diarization results for $test"
  [ ! -f $ref_rttm.scoring ] && sed 's/_U0[1-6]\.ENH//g' $ref_rttm > $ref_rttm.scoring
  [ ! -f $hyp_rttm.scoring ] && sed 's/_U0[1-6]\.ENH//g' $hyp_rttm > $hyp_rttm.scoring
  ref_rttm_path=$(readlink -f ${ref_rttm}.scoring)
  hyp_rttm_path=$(readlink -f ${hyp_rttm}.scoring)
  [ ! -f ./local/uem_file.scoring ] && cat ./local/uem_file | grep 'U06' | sed 's/_U0[1-6]//g' > ./local/uem_file.scoring
  cd dscore && python score.py -u ../local/uem_file.scoring -r $ref_rttm_path \
    -s $hyp_rttm_path 2>&1 | tee -a ../$scoring/DER && cd .. || exit 1;
  touch $scoring/.done
fi
