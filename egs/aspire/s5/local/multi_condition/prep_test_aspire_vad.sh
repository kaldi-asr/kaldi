#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2015.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire 
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

set -u
set -o pipefail
set -e 

use_icsi_method=false
iter=final
mfccdir=mfcc_reverb_submission
stage=0
decode_num_jobs=200
num_jobs=30
LMWT=12
word_ins_penalty=0
min_lmwt=9
max_lmwt=20
word_ins_penalties=0.0,0.25,0.5,0.75,1.0
decode_mbr=true
acwt=0.1
lattice_beam=8
ctm_beam=6
do_segmentation=true
max_count=100 # parameter for extract_ivectors.sh
sub_speaker_frames=1500
overlap=5
window=30
affix=
ivector_scale=1.0
pad_frames=0  # this did not seem to be helpful but leaving it as an option.
tune_hyper=true
pass2_decode_opts=
filter_ctm=true
weights_file=
silence_weight=0.00001
create_whole_dir=true
use_vad_prob=false
use_lats=true
transform_weights=false
speech_to_sil_ratio=1
use_bootstrap_vad=false
nj=30
. cmd.sh

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <data-dir> <vad-model-dir> <lang-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet2_multicondition/nnet_ms_a"
  exit 1;
fi

data_dir=$1 #select from data/{dev_aspire,test_aspire,eval_aspire}
vad_model_dir=$2
lang=$3 # data/lang
dir=$4 # exp/nnet2_multicondition/nnet_ms_a

data_id=`basename $data_dir`
model_affix=`basename $dir`
ivector_dir=`dirname $dir`
ivector_affix=${affix:+_$affix}_$model_affix
vad_dir=exp/vad_${data_id}_${affix}
affix=_${affix}_iter${iter}
act_data_id=${data_id}
if [ "$data_id" == "test_aspire" ]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
  if [ $stage -le 0 ]; then
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf ${data_dir} exp/make_mfcc_reverb/${data_dir} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh $data_dir exp/make_mfcc_reverb/${data_dir} $mfccdir || exit 1;
  fi
elif [ "$data_id" == "eval_aspire" ]; then
  out_file=single_eval${affix}_$model_affix.ctm
  if [ $stage -le 0 ]; then
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf ${data_dir} exp/make_mfcc_reverb/${data_dir} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh $data_dir exp/make_mfcc_reverb/${data_dir} $mfccdir || exit 1;
  fi
else
  if $create_whole_dir; then
    if [ $stage -le 0 ]; then
      echo "Creating the data dir with whole recordings without segmentation"
      # create a whole directory without the segments
      unseg_dir=data/${data_id}_whole
      src_dir=data/$data_id
      mkdir -p $unseg_dir
      echo "Creating the $unseg_dir/wav.scp file"
      cp $src_dir/wav.scp $unseg_dir

      echo "Creating the $unseg_dir/reco2file_and_channel file"
      cat $unseg_dir/wav.scp | awk '{print $1, $1, "A";}' > $unseg_dir/reco2file_and_channel
      cat $unseg_dir/wav.scp | awk '{print $1, $1;}' > $unseg_dir/utt2spk
      utils/utt2spk_to_spk2utt.pl $unseg_dir/utt2spk > $unseg_dir/spk2utt

      steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf $unseg_dir exp/make_mfcc_reverb/${data_id}_whole $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh $unseg_dir exp/make_mfcc_reverb/${data_id}_whole $mfccdir || exit 1;
    fi
    data_id=${data_id}_whole
  fi
  out_file=single_dev${affix}_${model_affix}.ctm
fi

if [ $stage -le 1 ]; then
  echo "Generating uniform segments for VAD with length 600"
  mkdir -p ${vad_dir}
  rm -rf ${vad_dir}/data_uniform_windows600
  copy_data_dir.sh --validate-opts "--no-text" data/$data_id ${vad_dir}/data_uniform_windows600 || exit 1
  cp data/$data_id/reco2file_and_channel ${vad_dir}/data_uniform_windows600 || exit 1
  python local/multi_condition/create_uniform_segments.py --overlap 0 --window 600 ${vad_dir}/data_uniform_windows600 || exit 1
  for file in cmvn.scp feats.scp; do
    rm -f ${vad_dir}/data_uniform_windows600/$file
  done
  utils/validate_data_dir.sh --no-text --no-feats ${vad_dir}/data_uniform_windows600 || exit 1
fi
if [ $stage -le 2 ]; then
  diarization/prepare_data.sh --nj $nj --cmd "$train_cmd" ${vad_dir}/data_uniform_windows600 ${vad_dir} ${vad_dir}/mfcc || exit 1
fi

split_data.sh ${vad_dir}/data_uniform_windows600 $nj

noise_model_found=false
if [ -f $vad_model_dir/noise.11.mdl ]; then
  noise_model_found=true
fi

if [ $stage -le 3 ]; then
  if $noise_model_found; then
    $train_cmd JOB=1:$nj ${vad_dir}/do_vad.JOB.log \
      diarization/vad_gmm_3models.sh --config conf/vad_icsi_babel_3models.conf \
      --try-merge-speech-noise true --output-lattice $use_lats --write-feats true \
      --speech-to-sil-ratio $speech_to_sil_ratio --use-bootstrap-vad $use_bootstrap_vad \
      ${vad_dir}/data_uniform_windows600/split$nj/JOB \
      $vad_model_dir/silence.11.mdl $vad_model_dir/speech.11.mdl \
      $vad_model_dir/noise.11.mdl ${vad_dir}/JOB || exit 1
  else
    if ! $use_icsi_method; then
      $train_cmd JOB=1:$nj ${vad_dir}/do_vad.JOB.log \
        diarization/vad_gmm_2models.sh --config conf/vad_icsi_babel_3models.conf \
        --try-merge-speech-noise true --output-lattice $use_lats --write-feats true \
        --speech-to-sil-ratio $speech_to_sil_ratio \
        ${vad_dir}/data_uniform_windows600/split$nj/JOB \
        $vad_model_dir/silence.11.mdl $vad_model_dir/speech.11.mdl \
        ${vad_dir}/JOB || exit 1
    else
      $train_cmd JOB=1:$nj ${vad_dir}/do_vad.JOB.log \
        diarization/vad_gmm_icsi.sh --config conf/vad_icsi_babel.conf \
        --try-merge-speech-noise true --output-lattice $use_lats --write-feats true \
        --speech-to-sil-ratio $speech_to_sil_ratio \
        ${vad_dir}/data_uniform_windows600/split$nj/JOB \
        $vad_model_dir/silence.11.mdl $vad_model_dir/speech.11.mdl \
        ${vad_dir}/JOB || exit 1
    fi
  fi

  for n in `seq $nj`; do
    for x in `cat ${vad_dir}/data_uniform_windows600/split$nj/$n/utt2spk | awk '{print $1}'`; do
      cat ${vad_dir}/$n/$x.vad.final.scp
    done
  done | sort -k1,1 > ${vad_dir}/vad.scp
fi

segmented_data_dir=data/${data_id}_uniformsegmented_win${window}_over${overlap}

if [ $stage -le 4 ]; then
  if $use_bootstrap_vad || ! $use_vad_prob; then
    $train_cmd ${vad_dir}/get_vad_per_file.log \
      segmentation-to-rttm \
      --segments=${vad_dir}/data_uniform_windows600/segments \
      scp:${vad_dir}/vad.scp - \| grep SPEECH \| \
      rttmSort.pl \| diarization/convert_rttm_to_segments.pl \| \
      segmentation-init-from-segments - ark:${vad_dir}/vad_per_file.ark
  else 
    if $use_lats; then
      for n in `seq $nj`; do
        for x in `cat ${vad_dir}/data_uniform_windows600/split$nj/$n/utt2spk | awk '{print $1}'`; do
          cat ${vad_dir}/$n/$x.lat.scp
        done | tee ${vad_dir}/$n/lats.scp
      done | sort -k1,1 > ${vad_dir}/lats.scp

      $train_cmd JOB=1:$nj ${vad_dir}/log/get_vad_weights.JOB.log \
        lattice-to-post scp:${vad_dir}/JOB/lats.scp ark:- \| \
        post-to-pdf-post ${vad_dir}/JOB/trans.mdl ark:- ark:- \| \
        weight-pdf-post $silence_weight 0:2 ark:- ark:- \| \
        post-to-weights ark:- ark,t:- \| \
        copy-vector ark,t:- ark:${vad_dir}/weights.JOB.ark
    else
      for n in `seq $nj`; do
        for x in `cat ${vad_dir}/data_uniform_windows600/split$nj/$n/utt2spk | awk '{print $1}'`; do
          gmm-compute-likes ${vad_dir}/$n/$x.final.mdl ark:${vad_dir}/$n/$x.feat.ark ark:- | \
            loglikes-to-post ark:- ark:- | \
            weight-pdf-post $silence_weight 0:2 ark:- ark:- | \
            post-to-weights ark:- ark,t:- | \
            copy-vector ark,t:- ark:-
        done > ${vad_dir}/weights.$n.ark
      done 
    fi
  fi
fi

if [ $stage -le 5 ]; then
  echo "Generating uniform segments with length $window and overlap $overlap."
  rm -rf $segmented_data_dir
  copy_data_dir.sh --validate-opts "--no-text" data/$data_id $segmented_data_dir || exit 1;
  cp data/$data_id/reco2file_and_channel $segmented_data_dir/ || exit 1;
  python local/multi_condition/create_uniform_segments.py --overlap $overlap --window $window $segmented_data_dir  || exit 1;
  for file in cmvn.scp feats.scp; do
    rm -f $segmented_data_dir/$file
  done
  utils/validate_data_dir.sh --no-text --no-feats $segmented_data_dir || exit 1;
fi

segmented_data_id=`basename $segmented_data_dir`
if [ $stage -le 6 ]; then
  echo "Extracting features for the segments"
  # extract the features/i-vectors once again so that they are indexed by utterance and not by recording
  rm -rf data/${segmented_data_id}_hires
  copy_data_dir.sh --validate-opts "--no-text " data/${segmented_data_id} data/${segmented_data_id}_hires || exit 1;
  steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/${segmented_data_id}_hires \
    exp/make_reverb_hires/${segmented_data_id} $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/${segmented_data_id}_hires exp/make_reverb_hires/${segmented_data_id} $mfccdir || exit 1;
  utils/fix_data_dir.sh data/${segmented_data_id}_hires
  utils/validate_data_dir.sh --no-text data/${segmented_data_id}_hires
fi

if [ ! -z $weights_file ]; then
  echo "$0: Using provided weights file $weights_file"
  ivector_extractor_input=$weights_file
else
  if [ $stage -le 7 ]; then
    mkdir -p $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}
    $train_cmd $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/log/get_file_lengths.log \
      feat-to-len scp:data/${data_id}/feats.scp \
      ark,t:$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_lengths.ark

    if ! $use_vad_prob; then
      segmentation-to-ali --default-label=0 \
        --lengths=ark,t:$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_lengths.ark \
        ark:${vad_dir}/vad_per_file.ark ark,t:- | \
        perl -e '
      my $silence_weight = shift @ARGV;
      while (<STDIN>) {
        chomp;
        @A = split;
        $utt = shift @A;
        print STDOUT "$utt [";
        for ($i = 0; $i <= $#A; $i++) {
          if ($A[$i] == 0) {
            print STDOUT " $silence_weight";
          } else {
          print STDOUT " $A[$i]";
        }
      }
      print STDOUT " ]\n";
    }' $silence_weight | copy-vector ark,t:- "ark:| gzip -c > $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_weights.gz"
    else 
      weight_vecs=
      for n in `seq $nj`; do 
        weight_vecs="${weight_vecs}${vad_dir}/weights.$n.ark "
      done

      $train_cmd $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/log/get_vad_file_weights.log \
        combine-vector-segments --max-overshoot=2 "ark:cat $weight_vecs|" \
        ${vad_dir}/data_uniform_windows600/segments \
        ark,t:$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_lengths.ark \
        "ark:| gzip -c > $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_weights.gz"
    fi
  fi

  cat $segmented_data_dir/segments | awk '{print $1" "$2" "$3" "$4-0.02}' > $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/truncated_segments

  x_th=0.8
  if [ $stage -le 8 ]; then
    if $transform_weights; then
      $train_cmd $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/log/extract_weights.log \
        extract-vector-segments "ark:gunzip -c $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_weights.gz |" \
        $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/truncated_segments ark,t:- \| \
        awk -v x_th=$x_th '{printf $1" [ "; for(i=3;i<=NF-1;i++) printf 1/sqrt(1+2*exp(-20*($i-x_th)))" " ; print "]"}' \| \
        copy-vector ark,t:- "ark:| gzip -c >$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/weights.gz"
    else 
      $train_cmd $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/log/extract_weights.log \
        extract-vector-segments "ark:gunzip -c $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/file_weights.gz |" \
        $ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/truncated_segments \
        "ark:| gzip -c >$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/weights.gz"
    fi
  fi
  ivector_extractor_input=$ivector_dir/ivector_weights_${segmented_data_id}${ivector_affix}/weights.gz
fi

if [ $stage -le 9 ]; then
  echo "Extracting i-vectors, stage 2 with input $ivector_extractor_input"
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
  echo "Generating lattices, stage 2 with --acwt $acwt"
  local/multi_condition/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config $pass2_decode_opts \
    --skip-scoring true --iter $iter --acwt $acwt --lattice-beam $lattice_beam \
    --online-ivector-dir $ivector_dir/ivectors_${segmented_data_id}${ivector_affix} \
    exp/tri5a/graph_pp data/${segmented_data_id}_hires ${decode_dir}_tg || \
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
        $decode_cmd LMWT=$min_lmwt:$max_lmwt $decode_dir/scoring/log/score.LMWT.\$wip.log \
          local/multi_condition/get_ctm.sh --filter-ctm-command "$filter_ctm_command" \
            --window $window --overlap $overlap \
            --beam $ctm_beam --decode-mbr $decode_mbr \
            --glm data/${act_data_id}/glm --stm data/${act_data_id}/stm \
          LMWT \$wip $lang data/${segmented_data_id}_hires $model $decode_dir || exit 1; 
      
      local/multi_condition/get_ctm_conf.sh --cmd "$decode_cmd" \
        --use-segments true \
        data/${segmented_data_id}_hires \
        ${lang} \
        ${decode_dir} || exit 1;

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
  if [ "$act_data_id" == "test_aspire" ] || [ "$act_data_id" == "eval_aspire" ]; then
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

