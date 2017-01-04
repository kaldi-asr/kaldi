#!/bin/bash 

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2015  Guoguo Chen
#           2016  Hainan Xu

# This script trains LMs on the WSJ LM-training data.
# It requires that you have already run wsj_extend_dict.sh,
# to get the larger-size dictionary including all of CMUdict
# plus any OOVs and possible acronyms that we could easily 
# derive pronunciations for.

# This script takes no command-line arguments but takes the --cmd option.

# Begin configuration section.

num_words_in=30000
num_words_out=50000
hidden_dim=512

stage=-100
sos="<s>"
eos="</s>"
oos="<oos>"

max_param_change=20
num_iters=100

num_train_frames_combine=10000 # # train frames for the above.                  
num_frames_diagnostic=2000 # number of frames for "compute_prob" jobs  
num_archives=50

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=128

initial_learning_rate=0.004
final_learning_rate=0.0002
learning_rate_decline_factor=1.04

type=rnn
cmd=run.pl

. cmd.sh

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

outdir=data/rnnlm-sigmoid-$initial_learning_rate-$final_learning_rate-$learning_rate_decline_factor-$minibatch_size-$type
srcdir=data/local/dict_nosp_larger
mkdir -p $outdir

export PATH=$KALDI_ROOT/tools/$rnnlm_ver:$PATH

if [ ! -f $srcdir/cleaned.gz -o ! -f $srcdir/lexicon.txt ]; then
  echo "Expecting files $srcdir/cleaned.gz and $srcdir/wordlist.final to exist";
  echo "You need to run local/wsj_extend_dict.sh before running this script."
  exit 1;
fi

if [ $stage -le -4 ]; then
  cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $outdir/wordlist.all

# Get training data with OOV words (w.r.t. our current vocab) replaced with <UNK>.
  echo "Getting training data with OOV words replaced with <UNK> (train_nounk.gz)" 
  gunzip -c $srcdir/cleaned.gz | awk -v w=$outdir/wordlist.all \
    'BEGIN{while((getline<w)>0) v[$1]=1;}
    {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<UNK> ";print ""}'|sed 's/ $//g' \
    | gzip -c > $outdir/all.gz

  echo "Splitting data into train and validation sets."
  heldout_sent=10000
  gunzip -c $outdir/all.gz | head -n $heldout_sent > $outdir/dev.txt.0 # validation data
  gunzip -c $outdir/all.gz | tail -n +$heldout_sent | \
    shuf --random-source=$outdir/all.gz > $outdir/train.txt.0 # training data

# Get unigram counts from our training data, and use this to select word-list
# for RNNLM training; e.g. 10k most frequent words.  Rest will go in a class
# that we (manually, at the shell level) assign probabilities for words that
# are in that class.  Note: this word-list doesn't need to include </s>; this
# automatically gets added inside the rnnlm program.
# Note: by concatenating with $outdir/wordlist.all, we are doing add-one
# smoothing of the counts.

  cat $outdir/train.txt.0 $outdir/wordlist.all | sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1 -n -r | awk '{print $2,$1}' > $outdir/unigramcounts.txt

  echo $sos 0 > $outdir/wordlist.in
  echo $oos 1 >> $outdir/wordlist.in
  cat $outdir/unigramcounts.txt | head -n $num_words_in | awk '{print $1,1+NR}' >> $outdir/wordlist.in

  echo $eos 0 > $outdir/wordlist.out
  echo $oos 1 >> $outdir/wordlist.out

  cat $outdir/unigramcounts.txt | head -n $num_words_out | awk '{print $1,1+NR}' >> $outdir/wordlist.out

  cat $outdir/train.txt.0 | awk -v sos="$sos" -v eos="$eos" '{print sos,$0,eos}' > $outdir/train.txt
  cat $outdir/dev.txt.0   | awk -v sos="$sos" -v eos="$eos" '{print sos,$0,eos}' > $outdir/dev.txt

fi

cp $outdir/wordlist.all $outdir/wordlist.rnn

num_words_in=`wc -l $outdir/wordlist.in | awk '{print $1}'`
num_words_out=`wc -l $outdir/wordlist.out | awk '{print $1}'`
num_words_total=`wc -l $outdir/unigramcounts.txt  | awk '{print $1}'`

if [ $stage -le -3 ]; then
  echo Get Examples
  $cmd $outdir/log/get-egs.train.txt \
    rnnlm-get-egs $outdir/train.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/train.egs" &
  $cmd $outdir/log/get-egs.dev.txt \
    rnnlm-get-egs $outdir/dev.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/dev.egs"

  (
    echo Do split
    [ -d $outdir/splitted-text ] && rm $outdir/splitted-text -r
    [ -d $outdir/egs ] && rm $outdir/egs -r
    mkdir -p $outdir/splitted-text
    mkdir -p $outdir/egs
    split --number=l/$num_archives --numeric-suffixes=1 $outdir/train.txt $outdir/splitted-text/train.

    for i in `seq 1 $num_archives`; do
      j=$i
      while [ ! -f $outdir/splitted-text/train.$i ]; do
        i=0$i
      done
      [ "$i" != "$j" ] && mv $outdir/splitted-text/train.$i $outdir/splitted-text/train.$j.txt
      rnnlm-get-egs $outdir/splitted-text/train.$j.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/egs/train.$j.egs"
    done
  )

  wait

  $cmd $outdir/log/create_train_subset_combine.log \
     nnet3-subset-egs --n=$num_train_frames_combine ark:$outdir/train.egs \
     ark,t:$outdir/train.subset.egs &                           

  cat $outdir/dev.txt | shuf | head -n $num_frames_diagnostic > $outdir/dev.diag.txt
  cat $outdir/train.txt | shuf | head -n $num_frames_diagnostic > $outdir/train.diag.txt
  rnnlm-get-egs $outdir/dev.diag.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/dev.subset.egs"
  rnnlm-get-egs $outdir/train.diag.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/train_diagnostic.egs"
#  $cmd $outdir/log/create_train_subset_diagnostic.log \
#     nnet3-subset-egs --n=$num_frames_diagnostic ark:$outdir/dev.egs \
#     ark,t:$outdir/dev.subset.egs
  wait
fi

oos_ratio=`cat $outdir/dev.diag.txt | awk -v w=$outdir/wordlist.out 'BEGIN{while((getline<w)>0) v[$1]=1;}
                                                         {for(i=2;i<=NF;i++){sum++; if(v[$i] != 1) oos++}} END{print oos/sum}'`

ppl_oos_penalty=`echo $num_words_out $num_words_total $oos_ratio | awk '{print ($2-$1)^$3}'`

echo dev oos ratio is $oos_ratio
echo dev oos penalty is $ppl_oos_penalty

if [ $stage -le -2 ]; then
  echo Create nnet configs

  if [ "$type" == "rnn" ]; then
  cat > $outdir/config <<EOF
  input-node name=input dim=$num_words_in
  component name=first_affine type=NaturalGradientAffineComponent input-dim=$[$num_words_in+$hidden_dim] output-dim=$hidden_dim  
  component name=first_nonlin type=SigmoidComponent dim=$hidden_dim
  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words_out
  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_out

#Component nodes
  component-node name=first_affine component=first_affine  input=Append(input, IfDefined(Offset(first_renorm, -1)))
  component-node name=first_nonlin component=first_nonlin  input=first_affine
  component-node name=first_renorm component=first_renorm  input=first_nonlin
  component-node name=final_affine component=final_affine  input=first_renorm
  component-node name=final_log_softmax component=final_log_softmax input=final_affine
  output-node    name=output input=final_log_softmax objective=linear
EOF
  elif [ "$type" == "lstm" ]; then
    steps/rnnlm/make_lstm_configs.py \
      --splice-indexes "$splice_indexes " \
      --num-lstm-layers $num_lstm_layers \
      --feat-dim $num_words_in \
      --cell-dim $cell_dim \
      --hidden-dim $hidden_dim \
      --recurrent-projection-dim $recurrent_projection_dim \
      --non-recurrent-projection-dim $non_recurrent_projection_dim \
      --norm-based-clipping $norm_based_clipping \
      --clipping-threshold $clipping_threshold \
      --num-targets $num_words_out \
      --label-delay $label_delay \
     $outdir/configs || exit 1;
    cp $outdir/configs/layer1.config $outdir/config
  fi
fi

if [ $stage -le 0 ]; then
  nnet3-init --binary=false $outdir/config $outdir/0.mdl
fi

cat data/local/dict/lexicon.txt | awk '{print $1}' > $outdir/wordlist.all.1
cat $outdir/wordlist.in $outdir/wordlist.out | awk '{print $1}' > $outdir/wordlist.all.2
cat $outdir/wordlist.all.[12] | sort -u > $outdir/wordlist.all

cp $outdir/wordlist.all $outdir/wordlist.rnn
touch $outdir/unk.probs
#rm $outdir/wordlist.all.[12]

mkdir -p $outdir/log/
if [ $stage -le $num_iters ]; then
  start=1
#  if [ $stage -gt 1 ]; then
#    start=$stage
#  fi
  learning_rate=$initial_learning_rate

  this_archive=0
  for n in `seq $start $num_iters`; do
    this_archive=$[$this_archive+1]

    [ $this_archive -gt $num_archives ] && this_archive=1

    echo for iter $n, training on archive $this_archive, learning rate = $learning_rate
    [ $n -ge $stage ] && (

        $cuda_cmd $outdir/log/train.rnnlm.$n.log nnet3-train \
        --max-param-change=$max_param_change "nnet3-copy --learning-rate=$learning_rate $outdir/$[$n-1].mdl -|" \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$outdir/egs/train.$this_archive.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" $outdir/$n.mdl

        if [ $n -gt 0 ]; then
          $cmd $outdir/log/progress.$n.log \
            nnet3-show-progress --use-gpu=no $outdir/$[$n-1].mdl $outdir/$n.mdl \
            "ark:nnet3-merge-egs ark:$outdir/train_diagnostic.egs ark:-|" '&&' \
            nnet3-info $outdir/$n.mdl &
        fi

      t=`grep "^# Accounting" $outdir/log/train.rnnlm.$n.log | sed "s/=/ /g" | awk '{print $4}'`
      w=`wc -w $outdir/splitted-text/train.$this_archive.txt | awk '{print $1}'`
      speed=`echo $w $t | awk '{print $1/$2}'`
      echo Processing speed: $speed words per second

    )

    learning_rate=`echo $learning_rate | awk -v d=$learning_rate_decline_factor '{printf("%f", $1/d)}'`
    if (( $(echo "$final_learning_rate > $learning_rate" |bc -l) )); then
      learning_rate=$final_learning_rate
    fi

    [ $n -ge $stage ] && (

      $decode_cmd $outdir/log/compute_prob_train.rnnlm.$n.log \
        nnet3-compute-prob $outdir/$n.mdl ark:$outdir/train.subset.egs &
      $decode_cmd $outdir/log/compute_prob_valid.rnnlm.$n.log \
        nnet3-compute-prob $outdir/$n.mdl ark:$outdir/dev.subset.egs 

      ppl=`grep Overall $outdir/log/compute_prob_valid.rnnlm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo DEV PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty
    ) &
  done
  cp $outdir/$num_iters.mdl $outdir/rnnlm
fi

#./local/rnnlm/run-rescoring.sh --rnndir $outdir/ --id $id
