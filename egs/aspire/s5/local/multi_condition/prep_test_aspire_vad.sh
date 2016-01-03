#!/bin/bash
# Copyright 2015  Johns Hopkins University (Author: Daniel Povey), Vijayaditya Peddinti
#           2015  Vimal Manohar
# Apache 2.0.

# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

set -u
set -o pipefail
set -e

stage=-10

graph_dir=exp/tri5a/graph_pp
iter=final        # Acoustic model to be used for decoding
mfccdir=mfcc_reverb_submission    # Dir to store MFCC features
fbankdir=fbank_reverb_submission  # Dir to store Fbank features
sad_mfcc_config=conf/mfcc_hires.conf
sad_fbank_config=conf/fbank.conf
mfcc_config=conf/mfcc_hires.conf
fbank_config=conf/fbank.conf
add_frame_snr=true
append_to_orig_feats=false

nj=30             # number of parallel jobs for VAD and segmentation
decode_nj=200     # number of parallel jobs for decoding

# segmentation opts
segmentation_config=
segmentation_stage=-10
segmentation_method=Viterbi
quantization_bins=0:2.5:5:7.5:12.5
snr_predictor_iter=final
sad_model_iter=final

# ivector extraction opts
use_ivectors=false
max_count=100 # parameter for extract_ivectors.sh
sub_speaker_frames=1500
ivector_scale=1.0
weights_file=
weights_method=Viterbi
silence_weight=0

# Decoding and scoring opts
acwt=0.1
LMWT=12
word_ins_penalty=0
min_lmwt=9
max_lmwt=20
word_ins_penalties=0.0,0.25,0.5,0.75,1.0
decode_mbr=true

lattice_beam=8
ctm_beam=6
filter_ctm=true

# output opts
affix=                # append this to the directory names
create_whole_dir=true
tune_hyper=true

# stage opts
input_frame_snrs_dir=
input_vad_dir=

. cmd.sh

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <data-dir> <snr-predictor> <sad-model-dir> <lang-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet2_multicondition/nnet_ms_a"
  exit 1;
fi

data_dir=$1 #select from data/{dev_aspire,test_aspire,eval_aspire}*
snr_predictor=$2
sad_model_dir=$3
lang=$4 # data/lang
dir=$5 # exp/nnet2_multicondition/nnet_ms_a

data_id=`basename $data_dir`      # {dev,test,eval}_aspire*
model_affix=`basename $dir`       # nnet_ms_*
vad_affix=${affix:+_$affix}
ivector_dir=`dirname $dir`        # exp/nnet2_multicondition
ivector_affix=${affix:+_$affix}_$model_affix
frame_snrs_dir=exp/frame_snrs${vad_affix}_${data_id}
vad_dir=exp/vad_${data_id}${vad_affix} # Temporary directory for VAD
segmentation_dir=exp/segmentation_${data_id}${vad_affix}
affix=_${affix}_iter${iter}         # affix to be specific to AM used
act_data_id=${data_id}            # the original data_id before data_id gets
                                  # modified to something else

# Function to create mfcc features
make_mfcc () {
  if [ $# -lt 2 ] || [ $[$# % 2] -ne 0 ]; then
    echo "$0: make_mfcc: Not enough arguments. Some variable is probably not set"
    exit 1
  fi
  local this_nj=$nj
  local mfcc_config=$mfcc_config

  while [ $# -gt 0 ]; do
    if [ $[$# % 2] -ne 0 ]; then
      echo "$0: make_mfcc: Not enough arguments. Some variable is probably not set"
      exit 1
    fi
    case $1 in
      --nj)
        this_nj=$2
        shift; shift
        ;;
      --mfcc-config)
        mfcc_config=$2
        shift; shift
        ;;
      *)
        if [ $# -eq 2 ]; then
          break;
        else
          echo "$0: make_mfcc: Unknown arguments $*"
          exit 1
        fi
        ;;
    esac
  done

  if [ $# -ne 2 ]; then
    echo "$0: make_mfcc: Not enough arguments. Some variable is probably not set"
    exit 1
  fi

  local data_dir=$1
  local mfccdir=$2

  rm -rf ${data_dir}_hires
  utils/copy_data_dir.sh ${data_dir} ${data_dir}_hires
  [ -f $data_dir/reco2file_and_channel ] && cp $data_dir/reco2file_and_channel ${data_dir}_hires

  data_dir=${data_dir}_hires
  steps/make_mfcc.sh --nj $this_nj --cmd "$train_cmd" \
    --mfcc-config $mfcc_config \
    ${data_dir} exp/make_hires/${data_dir} $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh \
    ${data_dir} exp/make_hires/${data_dir} $mfccdir || exit 1;
  utils/fix_data_dir.sh ${data_dir}
  utils/validate_data_dir.sh --no-text ${data_dir}
}

make_fbank () {
  if [ $# -lt 2 ] || [ $[$# % 2] -ne 0 ]; then
    echo "$0: make_fbank: Not enough arguments. Some variable is probably not set"
    exit 1
  fi
  local this_nj=$nj
  local fbank_config=conf/fbank_hires.conf

  while [ $# -gt 0 ]; do
    if [ $[$# % 2] -ne 0 ]; then
      echo "$0: make_fbank: Not enough arguments. Some variable is probably not set"
      exit 1
    fi
    case $1 in
      --nj)
        this_nj=$2
        shift; shift
        ;;
      --fbank-config)
        fbank_config=$2
        shift; shift
        ;;
      *)
        if [ $# -eq 2 ]; then
          break;
        else
          echo "$0: make_fbank: Unknown arguments $*"
          exit 1
        fi
        ;;
    esac
  done

  if [ $# -ne 2 ]; then
    echo "$0: make_fbank: Not enough arguments. Some variable is probably not set"
    exit 1
  fi


  local data_dir=$1
  local fbankdir=$2

  rm -rf ${data_dir}_fbank
  utils/copy_data_dir.sh ${data_dir} ${data_dir}_fbank
  [ -f $data_dir/reco2file_and_channel ] && cp $data_dir/reco2file_and_channel ${data_dir}_fbank

  data_dir=${data_dir}_fbank
  steps/make_fbank.sh --nj $this_nj --cmd "$train_cmd" \
    --fbank-config $fbank_config \
    ${data_dir} exp/make_fbank/${data_dir} $fbankdir || exit 1;
  steps/compute_cmvn_stats.sh --fake \
    ${data_dir} exp/make_fbank/${data_dir} $fbankdir || exit 1;
  utils/fix_data_dir.sh ${data_dir}
  utils/validate_data_dir.sh --no-text ${data_dir}
}

if [[ "$data_id" =~ "test_aspire" ]]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
  if [ $stage -le 0 ]; then
    make_mfcc --nj $nj --mfcc-config $sad_mfcc_config $data_dir $mfccdir
    make_fbank --nj $nj --fbank-config $sad_fbank_config $data_dir $fbankdir
  fi
elif [[ "$data_id" =~ "eval_aspire" ]]; then
  out_file=single_eval${affix}_$model_affix.ctm
  if [ $stage -le 0 ]; then
    make_mfcc --nj $nj --mfcc-config $sad_mfcc_config $data_dir $mfcc_dir
    make_fbank --nj $nj --fbank-config $sad_fbank_config $data_dir $fbankdir
  fi
else
  if $create_whole_dir; then
    if [ $stage -le 0 ]; then
      echo "Creating the data dir with whole recordings without segmentation"
      # create a whole directory without the segments for the
      # purposes of recreating the eval setting on dev set
      whole_dir=data/${data_id}_whole   # unsegmented_dir
      mkdir -p $whole_dir
      cp $data_dir/wav.scp $whole_dir    # same as before
      cat $whole_dir/wav.scp | \
        awk '{print $1, $1, "A";}' > $whole_dir/reco2file_and_channel

      cat $whole_dir/wav.scp | awk '{print $1, $1;}' > $whole_dir/utt2spk
      utils/utt2spk_to_spk2utt.pl $whole_dir/utt2spk > $whole_dir/spk2utt

      make_mfcc --nj $nj --mfcc-config $sad_mfcc_config $whole_dir $mfccdir

      make_fbank --nj $nj --fbank-config $sad_fbank_config $whole_dir $fbankdir
    fi
    data_id=${data_id}_whole
  fi
  out_file=single_dev${affix}_${model_affix}.ctm
fi

if [ $stage -le 1 ]; then
  # Compute sub-band SNR
  local/snr/compute_frame_snrs.sh --cmd "$train_cmd" \
    --use-gpu no --nj $nj --iter $snr_predictor_iter \
    $snr_predictor \
    data/${data_id}_hires data/${data_id}_fbank \
    $frame_snrs_dir || exit 1
fi

compute_sad_opts=(--quantization-bins $quantization_bins --iter $sad_model_iter)

if [ ! -z "$input_frame_snrs_dir" ] && [ $stage -ge 2 ]; then
  frame_snrs_dir=$input_frame_snrs_dir
fi

if [ $stage -le 2 ]; then
  local/snr/create_snr_data_dir.sh --cmd "$train_cmd" --nj $nj --append-to-orig-feats $append_to_orig_feats --add-frame-snr $add_frame_snr \
    data/${data_id}_fbank $frame_snrs_dir exp/make_snr_data_dir/${data_id} snr_feats $frame_snrs_dir/${data_id}_snr || exit 1
fi

if [ $stage -le 3 ]; then
  local/snr/compute_sad.sh \
    --nj $nj --use-gpu yes "${compute_sad_opts[@]}" \
    --snr-data-dir $frame_snrs_dir/${data_id}_snr \
    $sad_model_dir $frame_snrs_dir ${vad_dir} || exit 1
fi

segmented_data_dir=data/${data_id}_seg${vad_affix}
segmented_data_id=`basename $segmented_data_dir`

if [ ! -z "$input_vad_dir" ] && [ $stage -ge 3 ]; then
  vad_dir=$input_vad_dir
fi

if [ $stage -le 4 ]; then
  local/snr/sad_to_segments.sh --cmd "$train_cmd" \
    --method $segmentation_method --stage $segmentation_stage ${segmentation_config:+--config $segmentation_config} \
    data/${data_id}_hires ${vad_dir} $segmentation_dir $segmented_data_dir
fi

[ -f $data_dir/reco2file_and_channel ] && cp $data_dir/reco2file_and_channel ${segmented_data_dir}

if [ $stage -le 5 ]; then
  make_mfcc --nj $nj --mfcc-config $mfcc_config \
    $segmented_data_dir $mfccdir
fi

if $use_ivectors; then
  if [ ! -z "$weights_file" ]; then
    echo "$0: Using provided weights file $weights_file"
    ivector_extractor_input=$weights_file
  else
    mkdir -p $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}

    if [ $stage -le 6 ]; then
      local/snr/get_weights_for_ivector_extraction.sh --cmd queue.pl --nj $nj \
        --method $weights_method ${segmentation_config:+--config $segmentation_config} \
        --silence-weight $silence_weight \
        ${segmented_data_dir} ${vad_dir} \
        $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}
      ivector_extractor_input=$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/weights.gz
    fi
  fi
fi

if $use_ivectors && [ $stage -le 9 ]; then
  echo "Extracting i-vectors, with weights from $ivector_extractor_input"
  # this does offline decoding, except we estimate the iVectors per
  # speaker, excluding silence (based on alignments from a GMM decoding), with a
  # different script.  This is just to demonstrate that script.
  # the --sub-speaker-frames is optional; if provided, it will divide each speaker
  # up into "sub-speakers" of at least that many frames... can be useful if
  # acoustic conditions drift over time within the speaker's data.
  steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj 20 \
    --silence-weight $silence_weight \
    --sub-speaker-frames $sub_speaker_frames --max-count $max_count \
    data/${segmented_data_id}_hires $lang $ivector_dir/extractor \
    $ivector_extractor_input $ivector_dir/ivectors_${segmented_data_id}${ivector_affix} || exit 1;
fi

decode_dir=$dir/decode_${segmented_data_id}${affix}_pp
if [ $stage -le 10 ]; then
  echo "Generating lattices, with --acwt $acwt"

  ivector_opts=(--online-ivector-dir "")
  if $use_ivectors; then
    ivector_opts=(--online-ivector-dir $ivector_dir/ivectors_${segmented_data_id}${ivector_affix})
  fi

  local/multi_condition/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
    --config conf/decode.config \
    --skip-scoring true --iter $iter --acwt $acwt --lattice-beam $lattice_beam \
    "${ivector_opts[@]}" \
    $graph_dir data/${segmented_data_id}_hires ${decode_dir}_tg || \
    { echo "$0: Error decoding";  exit 1; }
fi

if [ $stage -le 11 ]; then
  echo "Rescoring lattices"
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    --skip-scoring true \
    ${lang}_pp_test{,_fg} data/${segmented_data_id}_hires \
    ${decode_dir}_{tg,fg} || exit 1;
fi

# tune the LMWT and WIP
# make command for filtering the ctms
decode_dir=${decode_dir}_fg
if [ -z $iter ]; then
  model=$decode_dir/../final.mdl # assume model one level up from decoding dir.
else
  model=$decode_dir/../$iter.mdl
fi

mkdir -p $decode_dir/scoring
# create a python script to filter the ctm, for labels which are mapped
# to null strings in the glm or which are not accepted by the scoring server
python -c "
import sys, re
lines = map(lambda x: x.strip(), open('data/${act_data_id}/glm').readlines())
patterns = []
for line in lines:
  if re.search('=>', line) is not None:
    parts = re.split('=>', line.split('/')[0])
    if parts[1].strip() == '':
      patterns.append(parts[0].strip())
print '|'.join(patterns)
" > $decode_dir/scoring/glm_ignore_patterns || exit 1;

ignore_patterns=$(cat $decode_dir/scoring/glm_ignore_patterns)
echo "$0: Ignoring these patterns from the ctm ", $ignore_patterns
cat << EOF > $decode_dir/scoring/filter_ctm.py
import sys
file = open(sys.argv[1])
out_file = open(sys.argv[2], 'w')
ignore_set = "$ignore_patterns".split("|")
ignore_set.append("[noise]")
ignore_set.append("[laughter]")
ignore_set.append("[vocalized-noise]")
ignore_set.append("!SIL")
ignore_set.append("<unk>")
ignore_set.append("%hesitation")
ignore_set = set(ignore_set)
print ignore_set
for line in file:
  if line.split()[4] not in ignore_set:
    out_file.write(line)
out_file.close()
EOF

filter_ctm_command="python $decode_dir/scoring/filter_ctm.py "

if  $tune_hyper ; then
  if [ $stage -le 12 ]; then
    if [[ "$act_data_id" =~ "dev_aspire" ]]; then
      wip_string=$(echo $word_ins_penalties | sed 's/,/ /g')
      temp_wips=($wip_string)
      $decode_cmd WIP=1:${#temp_wips[@]} $decode_dir/scoring/log/score.wip.WIP.log \
        wips=\(0 $wip_string\) \&\& \
        wip=\${wips[WIP]} \&\& \
        echo \$wip \&\& \
        $decode_cmd LMWT=$min_lmwt:$max_lmwt \
        $decode_dir/scoring/log/score.LMWT.\$wip.log \
          local/multi_condition/get_ctm.sh \
          --filter-ctm-command "$filter_ctm_command" \
          --beam $ctm_beam --decode-mbr $decode_mbr \
          --glm data/${act_data_id}/glm --stm data/${act_data_id}/stm \
          LMWT \$wip $lang data/${segmented_data_id}_hires \
          $model $decode_dir || exit 1;

      #local/multi_condition/get_ctm_conf.sh --cmd "$decode_cmd" \
      #  --use-segments true \
      #  data/${segmented_data_id}_hires \
      #  ${lang} ${decode_dir} || exit 1;

      eval "grep Sum $decode_dir/score_{${min_lmwt}..${max_lmwt}}/penalty_{$word_ins_penalties}/*.sys"|utils/best_wer.sh 2>/dev/null
      eval "grep Sum $decode_dir/score_{${min_lmwt}..${max_lmwt}}/penalty_{$word_ins_penalties}/*.sys" | \
       utils/best_wer.sh 2>/dev/null | python -c "import sys, re
line = sys.stdin.readline()
file_name=line.split()[-1]
parts=file_name.split('/')
penalty = re.sub('penalty_','',parts[-2])
lmwt = re.sub('score_','', parts[-3])
lmfile=open('$decode_dir/scoring/bestLMWT','w')
lmfile.write(str(lmwt))
lmfile.close()
wipfile=open('$decode_dir/scoring/bestWIP','w')
wipfile.write(str(penalty))
wipfile.close()
" || exit 1;
        LMWT=$(cat $decode_dir/scoring/bestLMWT)
        word_ins_penalty=$(cat $decode_dir/scoring/bestWIP)
    fi
  fi
  if [[ "$act_data_id" =~ "test_aspire" ]] || [[ "$act_data_id" =~ "eval_aspire" ]]; then
    dev_decode_dir=$(echo $decode_dir|sed "s/test_aspire/dev_aspire_whole/g; s/eval_aspire/dev_aspire_whole/g")
    if [ -f $dev_decode_dir/scoring/bestLMWT ]; then
      LMWT=$(cat $dev_decode_dir/scoring/bestLMWT)
      echo "Using the bestLMWT $LMWT value found in  $dev_decode_dir"
    else
      echo "Unable to find the bestLMWT in the  dev decode dir $dev_decode_dir"
      echo "Keeping the default/user-specified value"
    fi
    if [ -f $dev_decode_dir/scoring/bestWIP ]; then
      word_ins_penalty=$(cat $dev_decode_dir/scoring/bestWIP)
      echo "Using the bestWIP $word_ins_penalty value found in  $dev_decode_dir"
    else
      echo "Unable to find the bestWIP in the  dev decode dir $dev_decode_dir"
      echo "Keeping the default/user-specified value"
    fi
  else
    echo "Using the default/user-specified values for LMWT and word_ins_penalty"
  fi
fi

# lattice to ctm conversion and scoring.
if [ $stage -le 13 ]; then
  echo "Generating CTMs with LMWT $LMWT and word insertion penalty of $word_ins_penalty"
  local/multi_condition/get_ctm.sh --filter-ctm-command "$filter_ctm_command" \
    --beam $ctm_beam --decode-mbr $decode_mbr \
    $LMWT $word_ins_penalty $lang data/${segmented_data_id}_hires $model $decode_dir 2>$decode_dir/scoring/finalctm.LMWT$LMWT.WIP$word_ins_penalty.log || exit 1;
fi

if [ $stage -le 14 ]; then
  cat $decode_dir/score_$LMWT/penalty_$word_ins_penalty/ctm.filt | awk '{split($1, parts, "-"); printf("%s 1 %s %s %s\n", parts[1], $3, $4, $5)}' > $out_file
  cat ${segmented_data_dir}_hires/wav.scp | awk '{split($1, parts, "-"); printf("%s\n", parts[1])}' > $decode_dir/score_$LMWT/penalty_$word_ins_penalty/recording_names
  python local/multi_condition/fill_missing_recordings.py $out_file $out_file.submission $decode_dir/score_$LMWT/penalty_$word_ins_penalty/recording_names
  echo "Generated the ctm @ $out_file.submission from the ctm file $decode_dir/score_${LMWT}/penalty_$word_ins_penalty/ctm.filt"
fi

