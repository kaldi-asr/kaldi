#!/bin/bash

train_text=data/sdm1/train/text
dev_text=data/sdm1/dev/text

num_words_in=10000
num_words_out=10000

stage=-100
sos="<s>"
eos="</s>"
oos="<oos>"

max_param_change=20
num_iters=30

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=256

initial_learning_rate=0.002
final_learning_rate=0.0001
learning_rate_decline_factor=1.1

num_lstm_layers=1
cell_dim=64
hidden_dim=256
recurrent_projection_dim=0
non_recurrent_projection_dim=64
norm_based_clipping=true
clipping_threshold=30
label_delay=0  # 5
splice_indexes=0

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

outdir=data/sdm1/lstm-$initial_learning_rate-$final_learning_rate-$learning_rate_decline_factor-$minibatch_size
srcdir=data/local/dict

set -e

mkdir -p $outdir

if [ $stage -le -4 ]; then
  cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $outdir/wordlist.all

  cat $train_text | awk -v w=$outdir/wordlist.all \
      'BEGIN{while((getline<w)>0) v[$1]=1;}
      {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
      | shuf --random-source=$train_text > $outdir/train.txt.0

  cat $dev_text | awk -v w=$outdir/wordlist.all \
      'BEGIN{while((getline<w)>0) v[$1]=1;}
      {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
      | shuf --random-source=$dev_text > $outdir/dev.txt.0

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

num_words_in=`wc -l $outdir/wordlist.in | awk '{print $1}'`
num_words_out=`wc -l $outdir/wordlist.out | awk '{print $1}'`

if [ $stage -le -3 ]; then
  rnnlm-get-egs $outdir/train.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:$outdir/egs
fi

if [ $stage -le -2 ]; then

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

fi

if [ $stage -le 0 ]; then
  nnet3-init --binary=false $outdir/configs/layer1.config $outdir/0.mdl
fi


cat data/local/dict/lexicon.txt | awk '{print $1}' > $outdir/wordlist.all.1
cat $outdir/wordlist.in $outdir/wordlist.out | awk '{print $1}' > $outdir/wordlist.all.2
cat $outdir/wordlist.all.[12] | sort -u > $outdir/wordlist.all
#rm $outdir/wordlist.all.[12]
cp $outdir/wordlist.all $outdir/wordlist.rnn
touch $outdir/unk.probs

mkdir -p $outdir/log/
if [ $stage -le $num_iters ]; then
  start=1
#  if [ $stage -gt 1 ]; then
#    start=$stage
#  fi
  learning_rate=$initial_learning_rate

  for n in `seq $start $num_iters`; do
    echo for iter $n, learning rate is $learning_rate
    [ $n -ge $stage ] && (
        $cuda_cmd $outdir/log/train.rnnlm.$n.log nnet3-train \
        --max-param-change=$max_param_change "nnet3-copy --learning-rate=$learning_rate $outdir/$[$n-1].mdl -|" \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$outdir/egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" $outdir/$n.mdl
    )

    learning_rate=`echo $learning_rate | awk -v d=$learning_rate_decline_factor '{printf("%f", $1/d)}'`
    if (( $(echo "$final_learning_rate > $learning_rate" |bc -l) )); then
      learning_rate=$final_learning_rate
    fi

    [ $n -ge $stage ] && (
      nw=`wc -l $outdir/wordlist.all | awk '{print $1 - 3}'` # <s>, </s>, <oos>
      nw=`wc -l $outdir/wordlist.all | awk '{print $1}'` # <s>, </s>, <oos>
#      nw=`wc -l data/sdm1/cued_rnn_ce_1/unigram.counts | awk '{print $1}'`
#      $decode_cmd $outdir/dev.ppl.$n.log rnnlm-eval $outdir/$n.mdl $outdir/wordlist.in $outdir/wordlist.out $outdir/dev.txt $outdir/dev-probs-iter-$n.txt
      echo $decode_cmd $outdir/dev.ppl.$n.log rnnlm-eval --num-words=$nw $outdir/$n.mdl $outdir/wordlist.in $outdir/wordlist.out $outdir/dev.txt $outdir/dev-probs-iter-$n.txt
      $decode_cmd $outdir/dev.ppl.$n.log rnnlm-eval --num-words=$nw $outdir/$n.mdl $outdir/wordlist.in $outdir/wordlist.out $outdir/dev.txt $outdir/dev-probs-iter-$n.txt
      nw=`cat $outdir/dev.txt | awk '{a+=NF-1}END{print a}' `
      to_cost=`cat $outdir/dev-probs-iter-$n.txt | awk '{a+=$1}END{print -a}'`
      ppl=`echo $to_cost $nw | awk '{print exp($1/$2)}'`
      echo DEV PPL on model $n.mdl is $ppl | tee $outdir/log/dev.ppl.$n.txt
    ) &
  done
  cp $outdir/$num_iters.mdl $outdir/rnnlm
fi

./local/rnnlm/run-rescoring.sh --rnndir $outdir/ --type lstm
