#!/bin/bash

train_text=data/sdm1/train/text
dev_text=data/sdm1/dev/text
outdir=rnnlm
num_words_in=10000
num_words_out=10100
hidden_dim=200
stage=0
sos="<s>"
eos="</s>"
oos="<oos>"
max_param_change=2.0
num_iter=3

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

set -x

mkdir -p $outdir

if [ $stage -le -4 ]; then
  cat $train_text | cut -d " " -f 2-  | sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1 -n -r | awk '{print $2,$1}' > $outdir/unigramcounts.txt

  echo $sos 0 > $outdir/wordlist.in
  echo $oos 1 >> $outdir/wordlist.in
  cat $outdir/unigramcounts.txt | head -n $num_words_in | awk '{print $1,1+NR}' >> $outdir/wordlist.in
  echo $eos 0 > $outdir/wordlist.out
  echo $oos 1 >> $outdir/wordlist.out
  cat $outdir/unigramcounts.txt | head -n $num_words_out | awk '{print $1,1+NR}' >> $outdir/wordlist.out


  cat $train_text | cut -d " " -f 2- | awk -v sos="$sos" -v eos="$eos" '{print sos,$0,eos}' > $outdir/train.txt
  cat $dev_text   | cut -d " " -f 2- | awk -v sos="$sos" -v eos="$eos" '{print sos,$0,eos}' > $outdir/dev.txt

#  cat $train_text | cut -d " " -f 2- | awk -v f=$outdir/wordlist 'BEGIN{while((getline<f)>0) v[$1]=1}{for(i=1;i<=NF;i++){if(v[$i]==1)printf("%s ",$i);else printf("<unk> ")};print""}' > $outdir/train.txt
#  cat $dev_text   | cut -d " " -f 2- | awk -v f=$outdir/wordlist 'BEGIN{while((getline<f)>0) v[$1]=1}{for(i=1;i<=NF;i++){if(v[$i]==1)printf("%s ",$i);else printf("<unk> ")};print""}' > $outdir/dev.txt
fi

num_words_in=`wc -l $outdir/wordlist.in | awk '{print $1}'`
num_words_out=`wc -l $outdir/wordlist.out | awk '{print $1}'`

if [ $stage -le -3 ]; then
  rnnlm-get-egs $outdir/train.txt $outdir/wordlist.in $outdir/wordlist.out ark,b:$outdir/egs
#  rnnlm-get-egs rnnlm/train.txt.top rnnlm/wordlist.in rnnlm/wordlist.out "ark,t:| perl -pe 's/line/\nline/g' | grep . > rnnlm/egs"
#  rnnlm-get-egs rnnlm/train.txt.top rnnlm/wordlist.in rnnlm/wordlist.out "ark,t:rnnlm/egs"
fi

#  component name=first_affine type=NaturalGradientAffineComponent input-dim=$[$num_words_in+$hidden_dim] output-dim=$hidden_dim  bias-stddev=0 
#  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words_out  param-stddev=0 bias-stddev=0 
if [ $stage -le -2 ]; then
  cat > $outdir/config <<EOF
  input-node name=input dim=$num_words_in
  component name=first_affine type=NaturalGradientAffineComponent input-dim=$[$num_words_in+$hidden_dim] output-dim=$hidden_dim  
  component name=first_nonlin type=RectifiedLinearComponent dim=$hidden_dim
  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words_out
  component name=final_nonlin type=TanhComponent dim=$num_words_out
  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_out

#Component nodes
  component-node name=first_affine component=first_affine  input=Append(input, IfDefined(Offset(first_renorm, -1)))
  component-node name=first_nonlin component=first_nonlin  input=first_affine
  component-node name=first_renorm component=first_renorm  input=first_nonlin
  component-node name=final_affine component=final_affine  input=first_renorm
  component-node name=final_nonlin component=final_nonlin  input=final_affine
  component-node name=final_log_softmax component=final_log_softmax input=final_nonlin
  output-node    name=output input=final_log_softmax objective=linear
EOF

fi

if [ $stage -le 0 ]; then
  nnet3-init --binary=false $outdir/config $outdir/0.mdl
fi

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=1024 # This default is suitable for GPU-based training.
                    # Set it to 128 for multi-threaded CPU-based training.

mkdir -p $outdir/LOGs/
if [ $stage -le $num_iter ]; then
  for n in `seq $stage $num_iter`; do
    $cuda_cmd $outdir/LOGs/train.rnnlm.$n.log nnet3-train \
        --max-param-change=$max_param_change $outdir/$n.mdl \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$outdir/egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" $outdir/$[$n+1].mdl
#  $cuda_cmd $outdir/train.rnnlm.log nnet3-train --max-param-change=$max_param_change --use-gpu=no $outdir/1.raw ark:$outdir/egs $outdir/2.raw
  done
fi
