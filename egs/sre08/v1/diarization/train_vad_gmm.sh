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
speech_num_gauss=16
sil_num_gauss=4
num_iters=20
impr_thres=0.002
stage=-10
cleanup=true
select_top_frames=true
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

feat_dim=`feat-to-dim "scp:head -n 1 $data/feats.scp |" ark,t:- | awk '{print $2}'` || exit 1

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
  diarization/prepare_vad_lang.sh --num-sil-states 30 --num-nonsil-states 75 \
    $dir/local/dict $dir/local/lang $dir/lang || exit 1
  fstisstochastic $dir/lang/G.fst  || echo "[info]: G not stochastic."
fi

if [ $stage -le -1 ]; then 
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  
  diarization/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
fi
 
if [ $stage -le 0 ]; then
mkdir -p $dir/q
utils/split_data.sh $data $nj || exit 1

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

  if ! $select_top_frames; then
    gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=10 \
      "\$feats select-voiced-frames ark:- scp:$data/vad.scp ark:- |" \
      $dir/\$utt_id.speech.0.mdl || exit 1
    gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=6 \
      "\$feats select-voiced-frames --select-unvoiced-frames=true ark:- scp:$data/vad.scp ark:- |" \
      $dir/\$utt_id.silence.0.mdl || exit 1
  else
    gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=12 \
      "\$feats select-top-frames --top-frames-proportion=0.16 ark:- ark:- |" \
      $dir/\$utt_id.speech.0.mdl || exit 1
    gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=8 \
      "\$feats select-top-frames --bottom-frames-proportion=0.04 --top-frames-proportion=0.0 ark:- ark:- |" \
      $dir/\$utt_id.silence.0.mdl || exit 1
  fi

  {
    cat $dir/trans.mdl
    echo "<DIMENSION> $feat_dim <NUMPDFS> 2"
    gmm-global-copy --binary=false $dir/\$utt_id.silence.0.mdl -
    gmm-global-copy --binary=false $dir/\$utt_id.speech.0.mdl -
  } > $dir/\$utt_id.0.mdl || exit 1

  x=0
  while [ \$x -lt $num_iters ]; do
    gmm-decode-simple \
      --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
      $dir/\$utt_id.\$x.mdl $dir/graph/HCLG.fst \
      "\$feats" ark:/dev/null ark:$dir/\$utt_id.\$x.ali || exit 1

    gmm-acc-stats-ali \
      $dir/\$utt_id.\$x.mdl "\$feats" \
      ark:$dir/\$utt_id.\$x.ali - | \
      gmm-est $dir/\$utt_id.\$x.mdl - $dir/\$utt_id.\$[x+1].mdl \
      2>&1 | tee $dir/log/update.\$utt_id.\$x.log || exit 1

    objf_impr=\$(cat $dir/log/update.\$utt_id.\$x.log | grep "GMM update: Overall .* objective function" | perl -pe 's/.*GMM update: Overall (\S+) objective function .*/\$1/')
    
    if [ "\$(perl -e "if (\$objf_impr < $impr_thres) { print true; }")" == true ]; then
      break;
    fi

    x=\$[x+1]
  done

  rm -f $dir/\$utt_id.final.mdl 2>/dev/null || true
  cp $dir/\$utt_id.\$x.mdl $dir/\$utt_id.final.mdl 
  
  gmm-decode-simple \
    --allow-partial=true --word-symbol-table=$dir/graph/words.txt \
    $dir/\$utt_id.final.mdl $dir/graph/HCLG.fst \
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
