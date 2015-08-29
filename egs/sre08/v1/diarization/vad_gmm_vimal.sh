#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -u 
set -o pipefail

# Begin configuration section.
cmd=run.pl
stage=-1

# Decode options
speech_duration=75
sil_duration=30
impr_thres=0.002
cleanup=true
use_loglikes_hypothesis=false
use_latgen=false
map_opts=

## Features paramters
window_size=100                   # 1s
force_ignore_energy_opts=

## Phase 1 parameters
num_frames_init_silence=2000      # 20s - Lowest energy frames selected to initialize Silence GMM
num_frames_init_sound=10000       # 100s - Highest energy frames selected to initialize Sound GMM
num_frames_init_sound_next=2000   # 20s - Highest zero crossing frames selected to initialize Sound GMM
sil_num_gauss_init=2
sound_num_gauss_init=2
sil_max_gauss=2
sound_max_gauss=8
sil_gauss_incr=0
sound_gauss_incr=2
sil_frames_incr=2000
sound_frames_incr=10000
sound_frames_next_incr=2000
num_iters=5
min_sil_variance=1
min_sound_variance=0.01
min_speech_variance=0.001

## Phase 2 parameters
speech_num_gauss_init=6
sil_max_gauss_phase2=7
sound_max_gauss_phase2=18
speech_max_gauss_phase2=16
sil_gauss_incr_phase2=1
sound_gauss_incr_phase2=2
speech_gauss_incr_phase2=2
num_iters_phase2=5
window_size_phase2=10

## Phase 3 parameters
sil_num_gauss_init_phase3=2
speech_num_gauss_init_phase3=2
sil_max_gauss_phase3=5
speech_max_gauss_phase3=12
sil_gauss_incr_phase3=1
speech_gauss_incr_phase3=2
num_iters_phase3=7


# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: vad_gmm_vimal.sh <data> <init-silence-model> <init-speech-model> <dir>"
  echo " e.g.: vad_gmm_icsi.sh data/rt05_eval exp/librispeech_s5/vad_model/silence.0.mdl exp/librispeech_s5/vad_model/speech.0.mdl exp/vad_rt05_eval"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data=$1
init_silence_model=$2
init_speech_model=$3
dir=$4

mkdir -p $dir
tmpdir=$dir/phase1
phase2_dir=$dir/phase2
phase3_dir=$dir/phase3

mkdir -p $tmpdir
mkdir -p $phase2_dir
mkdir -p $phase3_dir

init_model_dir=`dirname $init_speech_model`

for f in $data/feats.scp $data/wav.scp $init_speech_model $init_silence_model; do
  [ ! -s $f ] && echo "$0: could not find $f or $f is empty" && exit 1
done 

ignore_energy_opts=`cat $init_model_dir/ignore_energy_opts` || exit 1

add_zero_crossing_feats=`cat $init_model_dir/add_zero_crossing_feats` || exit 1

zc_opts=
[ -f conf/zc_vad.conf ] && zc_opts="--config=conf/zc_vad.conf"

feat_dim=`feat-to-dim "scp:head -n 1 $data/feats.scp |" ark,t:- | awk '{print $2}'` || exit 1

# Prepare a lang directory
if [ $stage -le -2 ]; then
  mkdir -p $dir/local
  mkdir -p $dir/local/dict
  mkdir -p $dir/local/lm

  echo $sil_phone_list | \
    while IFS=' ' read phone; do
      echo $phone
    done > $dir/local/dict/silence_phones.txt

  echo "1" > $dir/local/dict/optional_silence.txt
  
  echo $speech_phone_list | \
    while IFS=' ' read phone; do
      echo $phone
    done > $dir/local/dict/nonsilence_phones.txt

  echo "$sil_phone_list $speech_phone_list" | \
    while IFS=' ' read phone; do
      echo $phone $phone
    done > $dir/local/dict/lexicon.txt

  echo -e "" > $dir/local/dict/extra_questions.txt

  mkdir -p $dir/lang

  # Training-time language model for VAD
  diarization/prepare_vad_lang.sh --num-sil-states 1 --num-nonsil-states 1 \
    $dir/local/dict $dir/local/lang $dir/lang || exit 1
  fstisstochastic $dir/lang/G.fst  || echo "[info]: G not stochastic."

  # Testing-time language model for VAD
  diarization/prepare_vad_lang.sh --num-sil-states 30 --num-nonsil-states 75 \
    $dir/local/dict $dir/local/lang $dir/lang_test || exit 1
fi

if [ $stage -le -1 ]; then 
  $cmd $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  $cmd $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang_test/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans_test.mdl || exit 1
  
  diarization/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
  diarization/make_vad_graph.sh --iter trans_test $dir/lang_test $dir $dir/graph_test || exit 1
fi
 
cat <<EOF > $dir/pdf_to_tid.map
0 1
1 3
EOF

if [ $stage -le 0 ]; then
mkdir -p $dir/q
utils/split_data.sh $data $nj || exit 1

map_est=
[ ! -z "$init_vad_model" ] && map_est="-map"
[ ! -z "$init_speech_model" ] && map_est="-map"

for n in `seq $nj`; do
  cat <<EOF > $dir/q/do_vad.$n.sh
set -e 
set -o pipefail
set -u

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

while IFS=$'\n' read line; do
  feats="ark:echo \$line | copy-feats scp:- ark:- |"
  utt_id=\$(echo \$line | awk '{print \$1}')

  if [ ! -z "$init_vad_model" ]; then
    cp $init_vad_model $dir/\$utt_id.0.mdl
  elif [ -z "$init_speech_model" ] || [ -z "$init_sil_model" ]; then
    if ! $select_top_frames; then
      gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=10 \
        "\$feats select-voiced-frames ark:- scp:$data/vad.scp ark:- |" \
        $dir/\$utt_id.speech.0.mdl || exit 1
      gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=6 \
        "\$feats select-voiced-frames --select-unvoiced-frames=true ark:- scp:$data/vad.scp ark:- |" \
        $dir/\$utt_id.silence.0.mdl || exit 1
    else
      gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=12 \
        "\$feats select-top-chunks --window-size=100 --top-frames-proportion=$top_frames_threshold ark:- ark:- |" \
        $dir/\$utt_id.speech.0.mdl || exit 1
      gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=8 \
        "\$feats select-top-chunks --window-size=100 --bottom-frames-proportion=$bottom_frames_threshold --top-frames-proportion=0.0 ark:- ark:- |" \
        $dir/\$utt_id.silence.0.mdl || exit 1
    fi

    {
      cat $dir/trans.mdl
      echo "<DIMENSION> $feat_dim <NUMPDFS> 2"
      gmm-global-copy --binary=false $dir/\$utt_id.silence.0.mdl -
      gmm-global-copy --binary=false $dir/\$utt_id.speech.0.mdl -
    } | gmm-copy - $dir/\$utt_id.0.mdl || exit 1
  else
    {
      cat $dir/trans.mdl
      echo "<DIMENSION> $feat_dim <NUMPDFS> 2"
      gmm-global-copy --binary=false $init_speech_model - 
      gmm-global-copy --binary=false $init_sil_model - 
    } | gmm-copy - $dir/\$utt_id.0.mdl || exit 1
  fi

  x=0
  while [ \$x -lt $num_iters ]; do
    if $use_loglikes_hypothesis; then
      gmm-compute-likes $dir/\$utt_id.\$x.mdl "\$feats" ark:- | \
        loglikes-to-post --min-post=$frame_select_threshold \
        ark:- "ark:| gzip -c > $dir/\$utt_id.\$x.post.gz" || exit 1

      gmm-acc-stats \
        $dir/\$utt_id.\$x.mdl "\$feats" \
        "ark:gunzip -c $dir/\$utt_id.\$x.post.gz | copy-post-mapped --id-map=$dir/pdf_to_tid.map ark:- ark:- |" - | \
        gmm-est${map_est} ${map_opts} --update-flags=mv $dir/\$utt_id.\$x.mdl - $dir/\$utt_id.\$[x+1].mdl \
        2>&1 | tee $dir/log/update.\$utt_id.\$x.log || exit 1
    elif $use_latgen; then
      gmm-latgen-faster --acoustic-scale=1.0 --determinize-lattice=false \
        $dir/\$utt_id.\$x.mdl $dir/graph/HCLG.fst \
        "\$feats" "ark:| gzip -c > $dir/\$utt_id.\$x.lat.gz" || exit 1

      lattice-to-post --acoustic-scale=1.0 \
        "ark:gunzip -c $dir/\$utt_id.\$x.lat.gz |" ark:- | \
        gmm-acc-stats $dir/\$utt_id.\$x.mdl "\$feats" ark:- - | \
        gmm-est${map_est} ${map_opts} --update-flags=mv $dir/\$utt_id.\$x.mdl - $dir/\$utt_id.\$[x+1].mdl \
        2>&1 | tee $dir/log/update.\$utt_id.\$x.log || exit 1
    else 
      gmm-decode-simple \
        --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
        $dir/\$utt_id.\$x.mdl $dir/graph/HCLG.fst \
        "\$feats" ark:/dev/null ark:$dir/\$utt_id.\$x.ali || exit 1

      gmm-acc-stats-ali \
        $dir/\$utt_id.\$x.mdl "\$feats" \
        ark:$dir/\$utt_id.\$x.ali - | \
        gmm-est${map_est} ${map_opts} --update-flags=mv $dir/\$utt_id.\$x.mdl - $dir/\$utt_id.\$[x+1].mdl \
        2>&1 | tee $dir/log/update.\$utt_id.\$x.log || exit 1
    fi

    objf_impr=\$(cat $dir/log/update.\$utt_id.\$x.log | grep "GMM update: Overall .* objective function" | perl -pe 's/.*GMM update: Overall (\S+) objective function .*/\$1/')
    
    if [ "\$(perl -e "if (\$objf_impr < $impr_thres) { print true; }")" == true ]; then
      break;
    fi

    x=\$[x+1]
  done

  rm -f $dir/\$utt_id.final.mdl 2>/dev/null || true
  #cp $dir/\$utt_id.\$x.mdl $dir/\$utt_id.final.mdl 

  (
  copy-transition-model --binary=false $dir/trans_test.mdl -
  gmm-copy --write-tm=false --binary=false $dir/\$utt_id.\$x.mdl -
  ) | gmm-copy - $dir/\$utt_id.final.mdl
  
  #gmm-decode-simple \
  #  --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
  #  $dir/\$utt_id.final.mdl $dir/graph/HCLG.fst \
  #  "\$feats" ark:/dev/null ark:$dir/\$utt_id.final.ali || exit 1
  
  gmm-decode-simple \
    --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
    $dir/\$utt_id.final.mdl $dir/graph_test/HCLG.fst \
    "\$feats" ark:/dev/null ark:$dir/\$utt_id.final.ali || exit 1
done < $data/split$nj/$n/feats.scp
EOF
done
fi

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/do_vad_job.JOB.log bash -x $dir/q/do_vad.JOB.sh || exit 1
fi

if $cleanup; then
  for x in `seq $[num_iters - 1]`; do
    if [ $[x % 10] -ne 0 ]; then
      rm $dir/*.$x.mdl
    fi
  done
fi

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

