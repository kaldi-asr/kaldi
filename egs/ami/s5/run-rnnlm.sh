#!/bin/bash

train_text=data/sdm1/train/text
dev_text=data/sdm1/dev/text
outdir=rnnlm
num_words=10000
hidden_dim=200
stage=1

. path.sh

mkdir -p $outdir

if [ $stage -lt 0 ]; then
  cat $train_text | cut -d " " -f 2-  | sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1 -n -r | awk '{print $2,$1}' > $outdir/unigramcounts.txt

  cat $outdir/unigramcounts.txt | head -n $num_words > $outdir/wordlist
  num_words=`wc -l $outdir/wordlist | awk '{print $1}'`

  cat $train_text | cut -d " " -f 2- | awk -v f=$outdir/wordlist 'BEGIN{while((getline<f)>0) v[$1]=1}{for(i=1;i<=NF;i++){if(v[$i]==1)printf("%s ",$i);else printf("<unk> ")};print""}' > $outdir/train.txt
  cat $dev_text   | cut -d " " -f 2- | awk -v f=$outdir/wordlist 'BEGIN{while((getline<f)>0) v[$1]=1}{for(i=1;i<=NF;i++){if(v[$i]==1)printf("%s ",$i);else printf("<unk> ")};print""}' > $outdir/dev.txt
fi

if [ $stage -le 1 ]; then
  cat > $outdir/config <<EOF
  input-node name=input dim=$num_words
  component name=first_affine type=NaturalGradientAffineComponent input-dim=$[$num_words+$hidden_dim] output-dim=$hidden_dim  bias-stddev=0 
  component name=first_nonlin type=RectifiedLinearComponent dim=$hidden_dim
  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words  param-stddev=0 bias-stddev=0 
  component name=final_nonlin type=TanhComponent dim=$num_words
  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words

#Component nodes
  component-node name=first_affine component=first_affine  input=Append(input, IfDefined(Offset(first_renorm, -1)))
  component-node name=first_nonlin component=first_nonlin  input=first_affine
  component-node name=first_renorm component=first_renorm  input=first_nonlin
  component-node name=final_affine component=final_affine  input=first_renorm
  component-node name=final_nonlin component=final_nonlin  input=final_affine
  component-node name=final_log_softmax component=final_log_softmax input=final_affine
  output-node    name=output input=final_log_softmax objective=linear
EOF

fi

if [ $stage -le 2 ]; then
  nnet3-init --binary=false $outdir/config $outdir/1.raw
fi
