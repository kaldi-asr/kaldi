#!/bin/bash

train_text=data/sdm1/cued_rnn_ce_1/train
dev_text=data/sdm1/cued_rnn_ce_1/valid
num_words_in=10000
num_words_out=10100
hidden_dim=200
stage=-100
sos="<s>"
eos="</s>"
oos="<oos>"
max_param_change=20
num_iter=20
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=64

initial_learning_rate=0.004
final_learning_rate=0.0004
learning_rate_decline_factor=1.2

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

outdir=rnnlm-sigmoid-$initial_learning_rate-$final_learning_rate-$learning_rate_decline_factor
srcdir=data/local/dict

#set -x
set -e

mkdir -p $outdir

if [ $stage -le -4 ]; then
  cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $outdir/wordlist.all

  cat $train_text $outdir/wordlist.all | sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1 -n -r | awk '{print $2,$1}' > $outdir/unigramcounts.txt

  echo $sos 0 > $outdir/wordlist.in
  echo $oos 1 >> $outdir/wordlist.in
  cat $outdir/unigramcounts.txt | head -n $num_words_in | awk '{print $1,1+NR}' >> $outdir/wordlist.in

  echo $eos 0 > $outdir/wordlist.out
  echo $oos 1 >> $outdir/wordlist.out

#  cp $outdir/wordlist.out $outdir/wordlist.rnn

  cat $outdir/unigramcounts.txt | head -n $num_words_out | awk '{print $1,1+NR}' >> $outdir/wordlist.out

  cat $train_text | awk -v sos="$sos" -v eos="$eos" '{print sos,$0,eos}' > $outdir/train.txt
  cat $dev_text   | awk -v sos="$sos" -v eos="$eos" '{print sos,$0,eos}' > $outdir/dev.txt
fi

num_words_in=`wc -l $outdir/wordlist.in | awk '{print $1}'`
num_words_out=`wc -l $outdir/wordlist.out | awk '{print $1}'`

if [ $stage -le -3 ]; then
  rnnlm-get-egs $outdir/train.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:$outdir/egs
#  rnnlm-get-egs rnnlm/train.txt.top rnnlm/wordlist.in rnnlm/wordlist.out "ark,t:| perl -pe 's/line/\nline/g' | grep . > rnnlm/egs"
#  rnnlm-get-egs rnnlm/train.txt.top rnnlm/wordlist.in rnnlm/wordlist.out "ark,t:rnnlm/egs"
fi

if [ $stage -le -2 ]; then
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
fi

if [ $stage -le 0 ]; then
  nnet3-init --binary=false $outdir/config $outdir/0.mdl
fi


cat data/local/dict/lexicon.txt | awk '{print $1}' > $outdir/wordlist.all.1
cat $outdir/wordlist.in $outdir/wordlist.out | awk '{print $1}' > $outdir/wordlist.all.2
cat $outdir/wordlist.all.[12] | sort -u > $outdir/wordlist.all
#rm $outdir/wordlist.all.[12]

mkdir -p $outdir/LOGs/
if [ $stage -le $num_iter ]; then
  start=1
#  if [ $stage -gt 1 ]; then
#    start=$stage
#  fi
  learning_rate=$initial_learning_rate

  for n in `seq $start $num_iter`; do
    echo for iter $n, learning rate is $learning_rate
    [ $n -ge $stage ] && (
        $cuda_cmd $outdir/LOGs/train.rnnlm.$n.log nnet3-train \
        --max-param-change=$max_param_change "nnet3-copy --learning-rate=$learning_rate $outdir/$[$n-1].mdl -|" \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$outdir/egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" $outdir/$n.mdl
    )


    learning_rate=`echo $learning_rate | awk -v d=$learning_rate_decline_factor '{print $1/d}'`
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
      echo DEV PPL on model $n.mdl is $ppl | tee $outdir/LOGs/dev.ppl.$n.txt
    ) &
  done
  cp $outdir/$num_iter.mdl $outdir/rnnlm
fi


#./run_rnnlms.sh
