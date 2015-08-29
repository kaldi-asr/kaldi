#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail

# Begin configuration section.
cmd=run.pl
nj=4
speech_duration=75
sil_duration=30
speech_max_gauss=12
sil_max_gauss=4
speech_gauss_incr=4
sil_gauss_incr=1
num_iters=20
impr_thres=0.002
stage=-10
cleanup=true
top_frames_threshold=0.16
bottom_frames_threshold=0.04
select_only_voiced_frames=false
ignore_energy_opts=
apply_cmvn=
init_speech_model=
init_silence_model=
window_size=3
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: diarization/train_vad_gmm.sh <data> <exp>"
  echo " e.g.: diarization/train_vad_gmm.sh data/dev exp/vad_dev"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi

data=$1
dir=$2

function build_0gram {
wordlist=$1; lm=$2
echo "=== Building zerogram $lm from ${wordlist}. ..."
awk '{print $1}' $wordlist | sort -u > $lm
python -c """
import math
with open('$lm', 'r+') as f:
 lines = f.readlines()
 p = math.log10(1/float(len(lines)));
 lines = ['%f\\t%s'%(p,l) for l in lines]
 f.seek(0); f.write('\\n\\\\data\\\\\\nngram  1=       %d\\n\\n\\\\1-grams:\\n' % len(lines))
 f.write(''.join(lines) + '\\\\end\\\\')
"""
}

for f in $data/feats.scp $data/vad.scp; do
  [ ! -s $f ] && echo "$0: could not find $f or $f is empty" && exit 1
done 

feat_dim=`feat-to-dim "ark:head -n 1 $data/feats.scp | add-deltas scp:- ark:- |$ignore_energy_opts" ark,t:- | awk '{print $2}'` || exit 1

# Prepare a lang directory
if [ $stage -le -2 ]; then
  mkdir -p $dir/local
  mkdir -p $dir/local/dict
  mkdir -p $dir/local/lm

  echo "1" > $dir/local/dict/silence_phones.txt
  echo "1" > $dir/local/dict/optional_silence.txt
  echo "2" > $dir/local/dict/nonsilence_phones.txt
  echo -e "1 1\n2 2" > $dir/local/dict/lexicon.txt
  echo -e "1\n2\n1 2" > $dir/local/dict/extra_questions.txt

  mkdir -p $dir/lang
  diarization/prepare_vad_lang.sh --num-sil-states 1 --num-nonsil-states 1 \
    $dir/local/dict $dir/local/lang $dir/lang || exit 1
  fstisstochastic $dir/lang/G.fst  || echo "[info]: G not stochastic."
  diarization/prepare_vad_lang.sh --num-sil-states 30 --num-nonsil-states 75 \
    $dir/local/dict $dir/local/lang $dir/lang_test || exit 1
fi

if [ $stage -le -1 ]; then 
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang_test/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans_test.mdl || exit 1
  
  diarization/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
  diarization/make_vad_graph.sh --iter trans_test $dir/lang_test $dir $dir/graph_test || exit 1
fi
 
cat <<EOF > $dir/pdf_to_tid.map
0 1
1 3
EOF

apply_cmvn_opts=
if $apply_cmvn; then
  apply_cmvn_opts="apply-cmvn-sliding ark:- ark:- |"
fi

if [ $stage -le 0 ]; then
mkdir -p $dir/q
utils/split_data.sh $data $nj || exit 1

select_frames_opts=
select_sil_frames_opts=
if $select_only_voiced_frames; then
  select_frames_opts="select-voiced-frames ark:- scp:$data/vad.scp ark:- |"
  select_sil_frames_opts="select-voiced-frames --select-unvoiced-frames=true ark:- scp:$data/vad.scp ark:- |"
fi

extract-column scp:$data/feats.scp ark,scp:mfcc/log_energies.ark,$data/log_energies.scp || { echo "extract-column failed"; exit 1; }

for n in `seq $nj`; do
  cat <<EOF > $dir/q/do_vad.$n.sh
set -e 
set -o pipefail
set -u

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

while IFS=$'\n' read line; do
  feats="ark:echo \$line | copy-feats scp:- ark:- | add-deltas ark:- ark:- |"
  utt_id=\$(echo \$line | awk '{print \$1}')
  echo \$utt_id > $dir/\$utt_id.list

  speech_num_gauss=6
  sil_num_gauss=2
  this_top_frames_threshold=$top_frames_threshold
  this_bottom_frames_threshold=$bottom_frames_threshold

  speech_energies="ark:utils/filter_scp.pl $dir/\$utt_id.list $data/log_energies.scp | copy-vector scp:- ark:- |"
  sil_energies="ark:utils/filter_scp.pl $dir/\$utt_id.list $data/log_energies.scp | copy-vector scp:- ark:- |"

  if [ -z "$init_speech_model" ]; then
    gmm-global-init-from-feats --num-gauss=\$speech_num_gauss --num-iters=4 \
      "\$feats $apply_cmvn_opts $select_frames_opts select-top-chunks --window-size=$window_size --frames-proportion=$top_frames_threshold --weights=\"\$speech_energies\" ark:- ark:- |$ignore_energy_opts" \
      $dir/\$utt_id.speech.0.mdl || exit 1
    #gmm-global-init-from-feats --num-gauss=\$sil_num_gauss --num-iters=4 \
    #  "\$feats $apply_cmvn_opts $select_sil_frames_opts select-top-chunks --window-size=$window_size --frames-proportion=$bottom_frames_threshold --select-frames=\$[sil_num_gauss * 20] --select-bottom-frames=true --weights=\"\$sil_energies\" ark:- ark:- |$ignore_energy_opts" \
    #  $dir/\$utt_id.silence.0.mdl || exit 1
    gmm-global-init-from-feats --num-gauss=\$sil_num_gauss --num-iters=4 \
      "\$feats $apply_cmvn_opts $select_sil_frames_opts select-top-chunks --window-size=$window_size --frames-proportion=$bottom_frames_threshold --select-bottom-frames=true --weights=\"\$sil_energies\" ark:- ark:- |$ignore_energy_opts" \
      $dir/\$utt_id.silence.0.mdl || exit 1
  else 
    gmm-global-get-frame-likes $init_speech_model \
      "\${feats}${apply_cmvn_opts}$ignore_energy_opts" ark:$dir/\$utt_id.speech_likes.init.ark || exit 1

    gmm-global-get-frame-likes $init_silence_model \
      "\${feats}${apply_cmvn_opts}$ignore_energy_opts" ark:$dir/\$utt_id.silence_likes.init.ark || exit 1

    loglikes-to-class --weights=ark:$dir/\$utt_id.weights.init.ark \
      ark:$dir/\$utt_id.silence_likes.init.ark ark:$dir/\$utt_id.speech_likes.init.ark \
      ark:$dir/\$utt_id.vad.init.ark || exit 1
    
    speech_energies="ark:utils/filter_scp.pl $dir/\$utt_id.list $data/log_energies.scp | copy-vector scp:- ark:- |"
    sil_energies="ark:utils/filter_scp.pl $dir/\$utt_id.list $data/log_energies.scp | copy-vector scp:- ark:- |"
    
    #gmm-global-init-from-feats --num-gauss=\$speech_num_gauss --num-iters=4 \
    #  "\$feats $apply_cmvn_opts select-voiced-frames ark:- ark:$dir/\$utt_id.vad.init.ark ark:- | select-top-chunks --window-size=$window_size --frames-proportion=$top_frames_threshold --weights=\"\$speech_energies\" ark:- ark:- |$ignore_energy_opts" \
    #  $dir/\$utt_id.speech.0.mdl || exit 1
    #gmm-global-init-from-feats --num-gauss=\$sil_num_gauss --num-iters=4 \
    #  "\$feats $apply_cmvn_opts select-voiced-frames --select-unvoiced-frames=true ark:- ark:$dir/\$utt_id.vad.init.ark ark:- | select-top-chunks --window-size=$window_size --select-frames=\$[sil_num_gauss * 20] --select-bottom-frames=true --weights=\"\$sil_energies\" ark:- ark:- |$ignore_energy_opts" \
    #  $dir/\$utt_id.silence.0.mdl || exit 1

    #gmm-global-init-from-feats --num-gauss=\$speech_num_gauss --num-iters=4 \
    #  "\$feats $apply_cmvn_opts select-top-chunks --window-size=1 --frames-proportion=$top_frames_threshold \
    #  --weights=\"\$speech_energies\" --selection-mask=ark:$dir/\$utt_id.vad.init.ark ark:- ark:- |$ignore_energy_opts" \
    #  $dir/\$utt_id.speech.0.mdl || exit 1

    gmm-global-init-from-feats --num-gauss=\$sil_num_gauss --num-iters=4 \
      "\$feats $apply_cmvn_opts select-top-chunks --select-bottom-frames=true --invert-mask=true \
      --window-size=1 --select-frames=\$[sil_num_gauss * 100] \
      --weights=\"\$sil_energies\" --selection-mask=ark:$dir/\$utt_id.vad.init.ark ark:- ark:- |$ignore_energy_opts" \
      $dir/\$utt_id.silence.0.mdl || exit 1
  
  fi

  x=0
  while [ \$x -lt $num_iters ]; do
    if [ $stage -le \$x ]; then

      if [ \$speech_num_gauss -le $speech_max_gauss ]; then
        speech_num_gauss=\$[speech_num_gauss + $speech_gauss_incr]
      fi

      if [ \$sil_num_gauss -le $sil_max_gauss ]; then
        sil_num_gauss=\$[sil_num_gauss + $sil_gauss_incr]
      fi

      this_top_frames_threshold=0.9
      #this_bottom_frames_threshold=1.0
      #this_top_frames_threshold=\$(perl -e "if (\$this_top_frames_threshold < 0.5) { print \$this_top_frames_threshold * 2 } else { print 0.5 }")
      this_bottom_frames_threshold=\$(perl -e "if (\$this_bottom_frames_threshold < 0.8) { print \$this_bottom_frames_threshold * 2 } else { print \$this_bottom_frames_threshold }")


      gmm-global-get-frame-likes $dir/\$utt_id.speech.\$x.mdl \
        "\${feats}${apply_cmvn_opts}$ignore_energy_opts" ark:$dir/\$utt_id.speech_likes.\$x.ark || exit 1

      gmm-global-get-frame-likes $dir/\$utt_id.silence.\$x.mdl \
        "\${feats}${apply_cmvn_opts}$ignore_energy_opts" ark:$dir/\$utt_id.silence_likes.\$x.ark || exit 1

      loglikes-to-class --weights=ark:$dir/\$utt_id.weights.\$x.ark \
        ark:$dir/\$utt_id.silence_likes.\$x.ark ark:$dir/\$utt_id.speech_likes.\$x.ark \
        ark:$dir/\$utt_id.vad.\$x.ark || exit 1

      speech_energies="ark:utils/filter_scp.pl $dir/\$utt_id.list $data/log_energies.scp | select-voiced-frames scp:- ark:$dir/\$utt_id.vad.\$x.ark ark:- |"
      sil_energies="ark:utils/filter_scp.pl $dir/\$utt_id.list $data/log_energies.scp | select-voiced-frames --select-unvoiced-frames=true scp:- ark:$dir/\$utt_id.vad.\$x.ark ark:- |"

      #gmm-global-init-from-feats --num-gauss=\$speech_num_gauss --num-iters=10 \
        #  "\$feats $apply_cmvn_opts select-top-chunks --window-size=$window_size --frames-proportion=\$this_top_frames_threshold --weights=scp:$data/log_energies.scp --selection-mask=ark:$dir/\$utt_id.vad.\$x.ark ark:- ark:- |$ignore_energy_opts" \
        #  $dir/\$utt_id.speech.\$[x+1].mdl || exit 1
      #

      #gmm-global-acc-stats $dir/\$utt_id.speech.\$x.mdl \
      #  "\$feats $apply_cmvn_opts select-voiced-frames ark:- ark:$dir/\$utt_id.vad.\$x.ark ark:- | select-top-chunks --window-size=$window_size --frames-proportion=\$this_top_frames_threshold --weights=\"\$speech_energies\" ark:- ark:- |$ignore_energy_opts" - | \
      #  gmm-global-est --mix-up=\$speech_num_gauss $dir/\$utt_id.speech.\$x.mdl \
      #  - $dir/\$utt_id.speech.\$[x+1].mdl || exit 1
      
      #gmm-global-init-from-feats --num-gauss=\$speech_num_gauss --num-iters=10 \
      #  "\$feats $apply_cmvn_opts select-top-chunks --window-size=$window_size --frames-proportion=\$this_top_frames_threshold \
      #  --weights=p:$data/log_energies.scp --selection-mask=ark:$dir/\$utt_id.vad.\$x.ark ark:- ark:- |$ignore_energy_opts" \
      #  $dir/\$utt_id.speech.\$[x+1].mdl || exit 1
      
      gmm-global-acc-stats $dir/\$utt_id.speech.\$x.mdl \
        "\$feats $apply_cmvn_opts select-top-chunks --window-size=$window_size --frames-proportion=\$this_top_frames_threshold \
        --weights=ark:$dir/\$utt_id.weights.\$x.ark --selection-mask=ark:$dir/\$utt_id.vad.\$x.ark ark:- ark:- |$ignore_energy_opts" - | \
        gmm-global-est --mix-up=\$speech_num_gauss $dir/\$utt_id.speech.\$x.mdl \
        - $dir/\$utt_id.speech.\$[x+1].mdl || exit 1
      

      #gmm-global-acc-stats $dir/\$utt_id.silence.\$x.mdl \
      #  "\$feats $apply_cmvn_opts select-voiced-frames --select-unvoiced-frames=true ark:- ark:$dir/\$utt_id.vad.\$x.ark ark:- | select-top-chunks --window-size=$window_size --select-frames=\$[sil_num_gauss * 40] --select-bottom-frames=true --weights=\"\$sil_energies\" ark:- ark:- |$ignore_energy_opts" - | \
      #  gmm-global-est --mix-up=\$sil_num_gauss $dir/\$utt_id.silence.\$x.mdl \
      #  - $dir/\$utt_id.silence.\$[x+1].mdl || exit 1
      
      gmm-global-acc-stats $dir/\$utt_id.silence.\$x.mdl \
        "\$feats $apply_cmvn_opts select-top-chunks --select-bottom-frames=true --invert-mask=true \
        --window-size=$window_size --select-frames=\$[sil_num_gauss * 100] \
        --weights=ark:$dir/\$utt_id.weights.\$x.ark --selection-mask=ark:$dir/\$utt_id.vad.\$x.ark ark:- ark:- |$ignore_energy_opts" - | \
        gmm-global-est --mix-up=\$sil_num_gauss $dir/\$utt_id.silence.\$x.mdl \
        - $dir/\$utt_id.silence.\$[x+1].mdl || exit 1

      #objf_impr=\$(cat $dir/log/update.\$utt_id.\$x.log | grep "GMM update: Overall .* objective function" | perl -pe 's/.*GMM update: Overall (\S+) objective function .*/\$1/')
      #
      #if [ "\$(perl -e "if (\$objf_impr < $impr_thres) { print true; }")" == true ]; then
      #  break;
      #fi
    fi
    x=\$[x+1]
  done

  rm -f $dir/\$utt_id.final.mdl 2>/dev/null || true
  #cp $dir/\$utt_id.\$x.mdl $dir/\$utt_id.final.mdl 

  (
  copy-transition-model --binary=false $dir/trans_test.mdl -
  echo "<DIMENSION> $feat_dim <NUMPDFS> 2"
  gmm-global-copy --binary=false $dir/\$utt_id.silence.\$x.mdl -
  gmm-global-copy --binary=false $dir/\$utt_id.speech.\$x.mdl -
  ) | gmm-copy - $dir/\$utt_id.final.mdl

  gmm-decode-simple \
    --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
    $dir/\$utt_id.final.mdl $dir/graph_test/HCLG.fst \
    "\${feats}${apply_cmvn_opts}$ignore_energy_opts" ark:/dev/null ark:$dir/\$utt_id.final.ali || exit 1
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

