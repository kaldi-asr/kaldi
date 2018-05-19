#!/bin/bash

. ./cmd.sh
. ./path.sh


cmd=queue.pl
stage=0
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
ppl_from_transcription=false
ppl_from_lattice=true
run_rescore=true
two_spks_mode=true
one_best_mode=false

# context_weight=5.0
# range=8
lm_weight=0.8
correction_weight=0.75

. ./utils/parse_options.sh

wordlist=data/lang/words.txt
ac_model_dir=exp/chain/tdnn_lstm_1e_sp
rnnlm_dir=exp/rnnlm_lstm_1c
text_dir=data/rnnlm_cache_adapt
mkdir -p $text_dir

if [ $stage -le 0 ]; then
  for dataset in eval2000 rt03; do
    data_dir=$text_dir/$dataset
    mkdir -p $data_dir 
    cat data/train/text | cut -d " " -f2- > $data_dir/train.txt
    cat $data_dir/train.txt $wordlist | sym2int.pl --map-oov "<unk>" $wordlist | \
      sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1nr  | awk '{print $2, $1}' > $data_dir/train.unigram
    cat data/$dataset/text | cut -d " " -f2- | tr A-Z a-z > $data_dir/$dataset.raw
    cat data/$dataset/text | cut -d " " -f1 > $data_dir/$dataset.head
    paste $data_dir/$dataset.head $data_dir/$dataset.raw | \
      sym2int.pl -f 2- --map-oov "<unk>" $wordlist > $data_dir/$dataset.txt
  done
fi

# compute perplexity by RNNLM adapted by a unigram cache model 
# estimated from trainscription
if [ $stage -le 1 ] && $ppl_from_transcription; then
  word_embedding="rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|"
  for dataset in eval2000 rt03; do 
    data_dir=$text_dir/$dataset
    echo Compute PPL from the adjusted RNNLM on $dataset...
    rnnlm-nbest-probs-adjust --correction-weight=$correction_weight \
      --two-speaker-mode=$two_spks_mode \
      $(cat $rnnlm_dir/special_symbol_opts.txt) \
      $rnnlm_dir/final.raw "$word_embedding" $data_dir/$dataset.txt \
      data/$dataset/utt2spk $data_dir/train.unigram
  done
fi

# compute perplexity by RNNLM adapted by a unigram cache model estimated
# from first pass decoded lattices
if [ $stage -le 2 ] && $ppl_from_lattice; then
  word_embedding="rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|"
  LM=fsh_sw1_fg
  for dataset in eval2000 rt03; do 
    data_dir=$text_dir/$dataset
    # decode_dir=${ac_model_dir}/decode_${dataset}_$LM
    decode_dir=${ac_model_dir}/new_split_${dataset}
    ppl_name=ppl_cache
    mkdir -p $data_dir/$ppl_name/log
    nj=`cat $decode_dir/num_jobs` || exit 1;

    echo Compute PPL from the adjusted RNNLM by lattice posteriors on $dataset...
    $cmd JOB=1:$nj $data_dir/$ppl_name/log/cw$correction_weight/perplexity.JOB.log \
    rnnlm-nbest-probs-adjust-lattice --correction-weight=$correction_weight \
      --lm-scale=$lm_weight \
      --two-speaker-mode=$two_spks_mode \
      --one-best-mode=$one_best_mode \
      $(cat $rnnlm_dir/special_symbol_opts.txt) \
      $rnnlm_dir/final.raw "$word_embedding" $data_dir/$dataset.txt \
      data/$dataset/utt2spk $data_dir/train.unigram \
      "ark:gunzip -c $decode_dir/lat.JOB.gz|"

    # Compute perplexity
    [ -f $data_dir/$ppl_name/log/cw$correction_weight/ppls.log ] &&
      rm $data_dir/$ppl_name/log/cw$correction_weight/ppls.log
    # $dataset.txt contains all words including eos (end of sentence symbol)
    word_count=`cat $data_dir/$dataset.txt | wc -w` 
    for i in `seq 1 $nj`; do
      grep 'Log' $data_dir/$ppl_name/log/cw$correction_weight/perplexity.$i.log | \
        awk '{n +=$NF}; END{print n}' >> $data_dir/$ppl_name/log/cw$correction_weight/ppls.log
    done
    awk '{n +=$1}; END{print n}' $data_dir/$ppl_name/log/cw$correction_weight/ppls.log \
      > $data_dir/$ppl_name/log/cw$correction_weight/ppls_sum.log
    logprobs=`cat $data_dir/$ppl_name/log/cw$correction_weight/ppls_sum.log`
    echo "scale=3;$logprobs/$word_count"|bc > \
      $data_dir/$ppl_name/log/cw$correction_weight/entropy.log
    ppl=`awk '{printf("%.1f",exp(-$1))}' $data_dir/$ppl_name/log/cw$correction_weight/entropy.log`
    echo "PPL by lattice posteriors on $dataset is $ppl" > \
      $data_dir/$ppl_name/log/cw$correction_weight/ppl
    echo "PPL by lattice posteriors on $dataset is $ppl"
  done
fi
exit 1;

if [ $stage -le 3 ] && $run_rescore; then
  LM=fsh_sw1_fg
  decode_out_dir=exp/chain/cache
  mkdir -p $decode_out_dir
  for decode_set in eval2000 rt03; do
    echo Perform pruned lattice-rescoring on $ac_model_dir on dataset $decode_set
    decode_dir=${ac_model_dir}/decode_${decode_set}_$LM
    decode_out=$decode_out_dir/decode_${decode_set}_${LM}_lmw${lm_weight}_cw${correction_weight}_pruned
    mkdir -p $decode_out
    cp $decode_dir/../final.mdl $decode_out_dir/

    rnnlm/lmrescore_rnnlm_lat_pruned_cache_adapt.sh \
      --cmd "$decode_cmd --mem 4G -l hostname='[bc]*'" \
      --weight $lm_weight \
      --correction-weight $correction_weight \
      --max-ngram-order $ngram_order \
      --two-speaker-mode $two_spks_mode \
      --one-best-mode $one_best_mode \
      data/lang_$LM $rnnlm_dir \
      data/${decode_set}_hires ${decode_dir} \
      $decode_out data/${decode_set}/utt2spk \
      $text_dir/$decode_set/train.unigram
  done
fi
