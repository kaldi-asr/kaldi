#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2015.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire 
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

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
. ./cmd.sh

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet2_multicondition/nnet_ms_a"
  exit 1;
fi

data_dir=$1 #select from {dev_aspire, test_aspire, eval_aspire}
lang=$2 # data/lang
dir=$3 # exp/nnet2_multicondition/nnet_ms_a

model_affix=`basename $dir`
ivector_dir=`dirname $dir`
ivector_affix=${affix:+_$affix}_${model_affix}_iter$iter
affix=_${affix}_iter${iter}
act_data_dir=${data_dir}
if [ "$data_dir" == "test_aspire" ]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
elif [ "$data_dir" == "eval_aspire" ]; then
  out_file=single_eval${affix}_$model_affix.ctm
else
  if [ $stage -le 1 ]; then
    echo "Creating the data dir with whole recordings without segmentation"
    # create a whole directory without the segments
    unseg_dir=data/${data_dir}_whole
    src_dir=data/$data_dir
    mkdir -p $unseg_dir
    echo "Creating the $unseg_dir/wav.scp file"
    cp $src_dir/wav.scp $unseg_dir

    echo "Creating the $unseg_dir/reco2file_and_channel file"
    cat $unseg_dir/wav.scp | awk '{print $1, $1, "A";}' > $unseg_dir/reco2file_and_channel
    cat $unseg_dir/wav.scp | awk '{print $1, $1;}' > $unseg_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $unseg_dir/utt2spk > $unseg_dir/spk2utt

    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf $unseg_dir exp/make_mfcc_reverb/${data_dir}_whole $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh $unseg_dir exp/make_mfcc_reverb/${data_dir}_whole $mfccdir || exit 1;
  fi
  data_dir=${data_dir}_whole
  out_file=single_dev${affix}_${model_affix}.ctm
fi

num_jobs=`cat data/${act_data_dir}/wav.scp|wc -l`
segmented_data_dir=${data_dir}
# extract the ivectors
if $do_segmentation; then
  segmented_data_dir=${data_dir}_uniformsegmented_win${window}_over${overlap}
fi

if [ $stage -le 2 ]; then
  echo "Generating uniform segments with length $window and overlap $overlap."
  rm -rf data/$segmented_data_dir
  copy_data_dir.sh --validate-opts "--no-text" data/$data_dir data/$segmented_data_dir || exit 1;
  cp data/$data_dir/reco2file_and_channel data/$segmented_data_dir/ || exit 1;
  python local/multi_condition/create_uniform_segments.py --overlap $overlap --window $window data/$segmented_data_dir  || exit 1;
  for file in cmvn.scp feats.scp; do
    rm -f data/$segmented_data_dir/$file
  done
  utils/validate_data_dir.sh --no-text --no-feats data/$segmented_data_dir || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "Extracting features for the segments"
   # extract the features/i-vectors once again so that they are indexed by utterance and not by recording
  rm -rf data/${segmented_data_dir}_hires
  copy_data_dir.sh --validate-opts "--no-text " data/${segmented_data_dir} data/${segmented_data_dir}_hires || exit 1;
  steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${segmented_data_dir}_hires \
      exp/make_reverb_hires/${segmented_data_dir} $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/${segmented_data_dir}_hires exp/make_reverb_hires/${segmented_data_dir} $mfccdir || exit 1;
  utils/fix_data_dir.sh data/${segmented_data_dir}_hires
  utils/validate_data_dir.sh --no-text data/${segmented_data_dir}_hires
fi

if [ $stage -le 4 ]; then
  echo "Extracting i-vectors, stage 1"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    --max-count $max_count \
    data/${segmented_data_dir}_hires $ivector_dir/extractor \
    $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}_stage1 || exit 1;
fi
if [ $ivector_scale != 1.0 ] && [ $ivector_scale != 1 ]; then
  ivector_scale_affix=_scale$ivector_scale
else
  ivector_scale_affix=
fi

if [ $stage -le 5 ]; then
  if [ "$ivector_scale_affix" != "" ]; then
    echo "$0: Scaling iVectors, stage 1"
    srcdir=$ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}_stage1
    outdir=$ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}${ivector_scale_affix}_stage1
    mkdir -p $outdir
    copy-matrix --scale=$ivector_scale scp:$srcdir/ivector_online.scp ark:- | \
      copy-feats --compress=true ark:-  ark,scp:$outdir/ivector_online.ark,$outdir/ivector_online.scp || exit 1;
    cp $srcdir/ivector_period $outdir/ivector_period
  fi
fi

decode_dir=$dir/decode_${segmented_data_dir}${affix}_pp
# generate the lattices
if [ $stage -le 6 ]; then
  echo "Generating lattices, stage 1"
  local/multi_condition/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config \
    --online-ivector-dir $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}${ivector_scale_affix}_stage1 \
    --skip-scoring true --iter $iter \
    exp/tri5a/graph_pp data/${segmented_data_dir}_hires ${decode_dir}_stage1 || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "$0: generating CTM from stage-1 lattices"
  local/multi_condition/get_ctm_conf.sh --cmd "$decode_cmd" \
    --use-segments false --iter $iter \
    data/${segmented_data_dir}_hires \
    ${lang} \
    ${decode_dir}_stage1 || exit 1;
fi

if [ $stage -le 8 ]; then
  if $filter_ctm; then
    if [ ! -z $weights_file ]; then
      echo "$0: Using provided weights file $weights_file"
      ivector_extractor_input=$weights_file
    else
      ctm=${decode_dir}_stage1/score_10/${segmented_data_dir}_hires.ctm 
      echo "$0: generating weights file from stage-1 ctm $ctm"
      
      feat-to-len scp:data/${segmented_data_dir}_hires/feats.scp ark,t:- >${decode_dir}_stage1/utt.lengths.$affix
      if [ ! -f $ctm ]; then  echo "$0: stage 8: expected ctm to exist: $ctm"; exit 1; fi
      cat $ctm | awk '$6 == 1.0 && $4 < 1.0' | \
      grep -v -w mm | grep -v -w mhm | grep -v -F '[noise]' | \
      grep -v -F '[laughter]' | grep -v -F '<unk>' | \
      perl -e ' $lengths=shift @ARGV;  $pad_frames=shift @ARGV; $silence_weight=shift @ARGV;
       $pad_frames >= 0 || die "bad pad-frames value $pad_frames";
       open(L, "<$lengths") || die "opening lengths file";
       @all_utts = ();
       $utt2ref = { };
       while (<L>) {
         ($utt, $len) = split(" ", $_);
         push @all_utts, $utt;
         $array_ref = [ ];
         for ($n = 0; $n < $len; $n++) { ${$array_ref}[$n] = $silence_weight; }
         $utt2ref{$utt} = $array_ref;
       }
       while (<STDIN>) {
         @A = split(" ", $_);
         @A == 6 || die "bad ctm line $_";
         $utt = $A[0]; $beg = $A[2]; $len = $A[3];
         $beg_int = int($beg * 100) - $pad_frames; 
         $len_int = int($len * 100) + 2*$pad_frames;
         $array_ref = $utt2ref{$utt};
         !defined $array_ref  && die "No length info for utterance $utt";
         for ($t = $beg_int; $t < $beg_int + $len_int; $t++) {
           if ($t >= 0 && $t < @$array_ref) {
             ${$array_ref}[$t] = 1;
            }
          }
        }
        foreach $utt (@all_utts) {  $array_ref = $utt2ref{$utt};
          print $utt, " [ ", join(" ", @$array_ref), " ]\n";
          } ' ${decode_dir}_stage1/utt.lengths.$affix $pad_frames $silence_weight   | gzip -c >${decode_dir}_stage1/weights${affix}.gz
          ivector_extractor_input=${decode_dir}_stage1/weights${affix}.gz
        fi
      else
        ivector_extractor_input=${decode_dir}_stage1
      fi
fi

if [ $stage -le 8 ]; then
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
    data/${segmented_data_dir}_hires $lang $ivector_dir/extractor \
    $ivector_extractor_input $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix} || exit 1;
fi

if [ $stage -le 9 ]; then
  echo "Generating lattices, stage 2 with --acwt $acwt"
  rm -f ${decode_dir}_tg/.error
  local/multi_condition/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config $pass2_decode_opts \
      --skip-scoring true --iter $iter --acwt $acwt --lattice-beam $lattice_beam \
      --online-ivector-dir $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix} \
     exp/tri5a/graph_pp data/${segmented_data_dir}_hires ${decode_dir}_tg || touch ${decode_dir}_tg/.error
  [ -f ${decode_dir}_tg/.error ] && echo "$0: Error decoding" && exit 1;
fi

if [ $stage -le 10 ]; then
  echo "Rescoring lattices"
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    --skip-scoring true \
    ${lang}_pp_test{,_fg} data/${segmented_data_dir}_hires \
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
lines = map(lambda x: x.strip(), open('data/${act_data_dir}/glm').readlines())
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
  if [ $stage -le 11 ]; then
    if [ "$act_data_dir" == "dev_aspire" ]; then
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
            --glm data/${act_data_dir}/glm --stm data/${act_data_dir}/stm \
          LMWT \$wip $lang data/${segmented_data_dir}_hires $model $decode_dir || exit 1; 

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
  if [ "$act_data_dir" == "test_aspire" ] || [ "$act_data_dir" == "eval_aspire" ]; then
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
if [ $stage -le 12 ]; then
  echo "Generating CTMs with LMWT $LMWT and word insertion penalty of $word_ins_penalty"
  local/multi_condition/get_ctm.sh --filter-ctm-command "$filter_ctm_command" \
    --beam $ctm_beam --decode-mbr $decode_mbr \
    $LMWT $word_ins_penalty $lang data/${segmented_data_dir}_hires $model $decode_dir 2>$decode_dir/scoring/finalctm.LMWT$LMWT.WIP$word_ins_penalty.log || exit 1;
fi

if [ $stage -le 13 ]; then
  cat $decode_dir/score_$LMWT/penalty_$word_ins_penalty/ctm.filt | awk '{split($1, parts, "-"); printf("%s 1 %s %s %s\n", parts[1], $3, $4, $5)}' > $out_file
  cat data/${segmented_data_dir}_hires/wav.scp | awk '{split($1, parts, "-"); printf("%s\n", parts[1])}' > $decode_dir/score_$LMWT/penalty_$word_ins_penalty/recording_names 
  local/multi_condition/fill_missing_recordings.py $out_file $out_file.submission $decode_dir/score_$LMWT/penalty_$word_ins_penalty/recording_names
  echo "Generated the ctm @ $out_file.submission from the ctm file $decode_dir/score_${LMWT}/penalty_$word_ins_penalty/ctm.filt"
fi
