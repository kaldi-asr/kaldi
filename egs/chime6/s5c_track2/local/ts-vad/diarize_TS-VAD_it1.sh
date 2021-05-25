#!/bin/bash
# Copyright   2020   Ivan Medennikov (STC-innovations Ltd)

# Apache 2.0.
#
# This script performs 1st iteration of TS-VAD diarization
# using an initial diarization rttm to estimate i-vectors

cmd="run.pl"
ref_rttm=
lang=data/lang

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
piece=10000 

ivector_affix=baseline-init

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 4 ]; then
  echo "Usage: $0 <ts-vad-dir> <ivector-dir> <initname> <out-dir>"
  echo "e.g.: $0 exp/ts-vad exp/nnet3 dev_beamformit_dereverb_diarized exp/ts-vad/it1"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --ref_rttm ./chime6_rttm/dev_rttm                # the location of the reference RTTM file"
  echo "  --ivector_affix baseline-init                    # affix corresponding to the initial diarization"
  echo "  --piece 10000                                    # raw wavs will be splitted into non-overlapping pieces of this size (in frames)"
  echo "  --thr 0.4                                        # post-processing: probability threshold"
  echo "  --window 51                                      # post-processing: median filter window (in frames)"
  echo "  --min_silence 0.3                                # post-processing: minimum length of silence (in seconds)"
  echo "  --min_speech 0.2                                 # post-processing: minimum length of speech (in seconds)"
  exit 1;
fi

dir=$1
ivector_dir=$2
initname=$3
outdir=$4

test="$(cut -d'_' -f1 <<<"$initname")"

#estimating i-vectors using the initial diarization
dset=${initname}_hires
ivdir=${ivector_dir}/ivectors_${dset}_${ivector_affix}
if [ ! -f $ivdir/ivector_online.scp ]; then
  echo "Extracting i-vectors for $dset"
  steps/online/nnet2/extract_ivectors.sh --cmd "$cmd" --nj $nj \
    --silence-weight 0.00001 \
    --sub-speaker-frames 0 --max-count 100 \
    data/$dset $lang $ivector_dir/extractor $ivdir || exit 1;
fi

#preparing 4-speaker track2 data
dsetsrc=$dset
name=$(echo $initname | sed s/_diarized//)
dset=${name}_U06_hires
if [ ! -f data/$dset/.done ]; then
  mkdir -p data/$dset
  cp data/$dsetsrc/wav.scp data/$dset/wav.scp
  awk '{print $1" "$1}' data/$dset/wav.scp > data/$dset/utt2spk
  awk '{print $1" "$1}' data/$dset/wav.scp > data/$dset/spk2utt
  utils/fix_data_dir.sh data/$dset
  steps/make_mfcc.sh --nj $nj_feats --mfcc-config conf/mfcc_hires.conf data/$dset data/$dset/log data/$dset/data || exit 1;
  touch data/$dset/.done
fi

#splitting 4-speaker track2 data into pieces
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

#preparing 4-speaker i-vectors
iv4dir=${ivector_dir}/ivectors-4spk_${dset}_${ivector_affix}
if [ ! -f $iv4dir/.done ]; then
  mkdir -p $iv4dir
  echo "Making pseudo-online 4spk i-vectors using source $ivdir"
  cat $ivdir/ivectors_spk.*.ark > $iv4dir/ivectors_spk.ark

  for spk in `seq 4`; do
    awk -v "spk=$spk" '{printf "%s %s-%s\n", $1, $2, spk}' data/$dset/utt2spk > data/$dset/utt2spk.$spk
  done

  $train_cmd JOB=1:4 $iv4dir/log/apply-map.JOB.log \
    utils/apply_map.pl -f 2 $iv4dir/ivectors_spk.ark \<data/$dset/utt2spk.JOB \>$iv4dir/ivectors_utt.JOB.ark || exit 1;

  ivector_dim=$[$(head -n 1 $ivdir/ivectors_spk.1.ark | wc -w) - 3] || exit 1;
  base_feat_dim=$(feat-to-dim scp:data/$dset/feats.scp -) || exit 1;
  start_dim=$base_feat_dim
  end_dim=$[$base_feat_dim+$ivector_dim-1]
  absdir=$(utils/make_absolute.sh $iv4dir)
  cp $ivdir/{ivector_period,final.ie.id} $iv4dir/
  ivector_period=$(cat $ivdir/ivector_period)

  $cmd JOB=1:4 $iv4dir/log/duplicate_feats.JOB.log \
    append-vector-to-feats scp:data/$dset/feats.scp ark:$iv4dir/ivectors_utt.JOB.ark ark:- \| \
    select-feats "$start_dim-$end_dim" ark:- ark:- \| \
    subsample-feats --n=$ivector_period ark:- ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$absdir/ivector_online.JOB.ark,$absdir/ivector_online.JOB.scp || exit 1;

  $cmd $iv4dir/log/paste-feats.log \
    paste-feats scp:$iv4dir/ivector_online.1.scp scp:$iv4dir/ivector_online.2.scp scp:$iv4dir/ivector_online.3.scp scp:$iv4dir/ivector_online.4.scp ark:- \| \
    copy-feats --compress=true ark:- ark,scp:$absdir/ivector_online.ark,$absdir/ivector_online.scp || exit 1;
  touch $iv4dir/.done
fi

#computing TS-VAD per-frame probabilities for each speaker
out=$outdir/$dset
if [ ! -f $out/.done ]; then
  local/ts-vad/compute_ts-vad_weights.sh --nj $nj_feats --use-gpu true --cmd "$cmd" --online-ivector-dir $iv4dir \
    --extra-left-context $extra_left_context --extra-right-context $extra_right_context --frames-per-chunk $frames_per_chunk \
    data/$dset $dir/final.raw $out || exit 1;
  touch $out/.done
fi

#TS-VAD probabilities post-processing and DER scoring
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
