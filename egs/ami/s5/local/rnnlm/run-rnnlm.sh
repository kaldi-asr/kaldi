#.!/bin/bash

train_text=data/sdm1/train/text
dev_text=data/sdm1/dev/text

cmd=run.pl

num_words_in=10000
num_words_out=10000
hidden_dim=200

stage=-100
sos="<s>"
eos="</s>"
oos="<oos>"

max_param_change=20
num_iters=40

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=64

num_train_frames_combine=10000 # # train frames for the above.                  
num_frames_diagnostic=2000 # number of frames for "compute_prob" jobs  
num_archives=9

initial_learning_rate=0.008
final_learning_rate=0.0004
learning_rate_decline_factor=1.05

type=rnn-newformat-40

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

outdir=data/sdm1/rnnlm-sigmoid-$initial_learning_rate-$final_learning_rate-$learning_rate_decline_factor-$minibatch_size-$type
srcdir=data/local/dict

set -e

mkdir -p $outdir

if [ $stage -le -4 ]; then
  echo Data Preparation
  cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $outdir/wordlist.all

#  cat $train_text | cut -d" " -f2- > $outdir/train.txt.0
#  cat $dev_text | cut -d" " -f2- > $outdir/dev.txt.0

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
  echo Get Examples
  $cmd $outdir/log/get-egs.train.txt \
    rnnlm-get-egs $outdir/train.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"| tr '\n' ' ' | sed 's=line=\nline=g' | grep . | shuf > $outdir/train.egs" &
  $cmd $outdir/log/get-egs.dev.txt \
    rnnlm-get-egs $outdir/dev.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"| tr '\n' ' ' | sed 's=line=\nline=g' | grep . > $outdir/dev.egs"

  wait

  [ -d $outdir/egs ] && rm $outdir/egs -r
  mkdir -p $outdir/egs

  echo Do split
  split --number=l/$num_archives --numeric-suffixes=1 $outdir/train.egs $outdir/egs/train.

  # delete leading zero's
  for i in `seq 1 $num_archives`; do
    j=$i
    while [ ! -f $outdir/egs/train.$i ]; do
      i=0$i
    done
    [ "$i" != "$j" ] && mv $outdir/egs/train.$i $outdir/egs/train.$j 
  done

  $cmd $outdir/log/create_train_subset_combine.log \
     nnet3-subset-egs --n=$num_train_frames_combine ark:$outdir/train.egs \
     ark,t:$outdir/train.subset.egs &                           
  $cmd $outdir/log/create_train_subset_diagnostic.log \
     nnet3-subset-egs --n=$num_frames_diagnostic ark:$outdir/dev.egs \
     ark,t:$outdir/dev.subset.egs
  wait
fi

if [ $stage -le -2 ]; then
  echo Create nnet configs
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

    echo for iter $n, learning rate is $learning_rate, training on archive $this_archive
    [ $n -ge $stage ] && (
        $cuda_cmd $outdir/log/train.rnnlm.$n.log nnet3-train \
        --max-param-change=$max_param_change "nnet3-copy --learning-rate=$learning_rate $outdir/$[$n-1].mdl -|" \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$outdir/egs/train.$this_archive ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" $outdir/$n.mdl
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
      echo DEV PPL on model $n.mdl is $ppl
    ) &
  done
  cp $outdir/$num_iters.mdl $outdir/rnnlm
fi

./local/rnnlm/run-rescoring.sh --rnndir $outdir/ --type $type
