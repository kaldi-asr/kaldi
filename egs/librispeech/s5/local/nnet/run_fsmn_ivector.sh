. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail
#########################

dnn_model=$1

stage=0

if [ $stage -le 0 ]; then
 min_seg_len=1.55
 train_set=train_960_cleaned
 gmm=tri6b_cleaned
 nnet3_affix=_cleaned
 local/nnet3/run_ivector_common.sh --stage $stage \
                                   --min-seg-len $min_seg_len \
                                   --train-set $train_set \
                                   --gmm $gmm \
                                   --num-threads-ubm 6 --num-processes 3 \
                                   --nnet3-affix "$nnet3_affix" || exit 1;
then

##Make fbank features
if [ $stage -le 1 ]; then
  mkdir -p data_fbank

  for x in train_960_cleaned test_other test_clean dev_other dev_clean; do
  fbankdir=fbank/$x
  
  cp -r data/$x data_fbank/$x
  steps/make_fbank.sh --nj 30 --cmd "$train_cmd"  --fbank-config conf/fbank.conf \
    data_fbank/$x exp/make_fbank/$x $fbankdir
  steps/compute_cmvn_stats.sh data_fbank/$x exp/make_fbank/$x $fbankdir
done
fi
###############
if [ $stage -le 2 ]; then

  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_960_cleaned data/lang exp/tri6b_cleaned exp/tri6b_cleaned_ali_train_960_cleaned
  steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
    data/dev_clean data/lang exp/tri6b_cleaned exp/tri6b_cleaned_ali_dev_clean
  steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
    data/dev_other data/lang exp/tri6b_cleaned exp/tri6b_cleaned_ali_dev_other
fi
#####CE-training
lrate=0.00001
dir=exp/tri7b_${dnn_model}
data_fbk=data_fbank
if [ $stage -le 3 ]; then
	proto=local/nnet/${dnn_model}.proto

        cat exp/nnet3_cleaned/ivectors_train_960_cleaned_hires/ivector_online.scp exp/nnet3_cleaned/ivectors_dev_clean_hires/ivector_online.scp \
            exp/nnet3_cleaned/ivectors_dev_other_hires/ivector_online.scp > exp/nnet3_cleaned/ivectors_train_960_dev_hires/ivector_online.scp

	$cuda_cmd $dir/_train_nnet.log \
   	steps/nnet/train_faster.sh --learn-rate $lrate --nnet-proto $proto \
        --start_half_lr 5 --momentum 0.9 \
	--train-tool "nnet-train-fsmn-streams" \
       	--feat-type plain --splice 1 \
	--cmvn-opts "--norm-means=true --norm-vars=false" --delta_opts "--delta-order=2" \
        --train-tool-opts "--minibatch-size=4096" \
        --ivector scp:exp/nnet3_cleaned/ivectors_train_960_dev_hires/ivector_online.scp \
	--ivector-append-tool "append-ivector-to-feats --online-ivector-period=10" \
       	$data_fbk/train_960_cleaned $data_fbk/dev_clean data/lang exp/tri6b_cleaned_ali_train_960_cleaned exp/tri6b_cleaned_ali_dev_clean $dir
fi
####Decode
acwt=0.08
if [ $stage -le 4 ]; then
	gmm=exp/tri6b_cleaned
	dataset="test_clean dev_clean test_other dev_other"
  	for set in $dataset
  	do
  	  	steps/nnet/decode.sh --nj 16 --cmd "$decode_cmd" \
		scoring_opts "--min-lmwt 10 --max-lmwt 30" \
      		--config conf/decode.config --acwt $acwt \
		$gmm/graph_tgsmall \
        	$data_fbk/$set $dir/decode_tgsmall_${set}

        	steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
        	$data_fbk/$set $dir/decode_{tgsmall,tgmed}_${set}
        	
		steps/lmrescore_const_arpa.sh \
		scoring_opts "--min-lmwt 10 --max-lmwt 30" \
        	--cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
       		$data_fbk/$set $dir/decode_{tgsmall,tglarge}_${set}
        	
		steps/lmrescore_const_arpa.sh \
		scoring_opts "--min-lmwt 10 --max-lmwt 30" \
        	--cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        	$data_fbk/$set $dir/decode_{tgsmall,fglarge}_${set}
  	done

	for set in $dataset; 
	do 
	for lm in fglarge tglarge tgmed tgsmall; 
	do 
		grep WER $dir/decode_${lm}_${set}*/wer* | ./utils/best_wer.sh 
	done
	done
fi

nj=32
if [ $stage -le 5 ]; then
        steps/nnet/align.sh --nj $nj --cmd "$train_cmd" $data_fbk/train_960_cleaned data/lang $dir ${dir}_ali
        steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
        $data_fbk/train_960_cleaned data/lang $dir ${dir}_denlats
fi

####do smbr
if [ $stage -le 5 ]; then
        steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --learn-rate 0.0000002 --acwt $acwt --do-smbr true \
        $data_fbk/train_960_cleaned data/lang $dir ${dir}_ali ${dir}_denlats ${dir}_smbr
fi

###decode
dir=${dir}_smbr
acwt=0.03
if [ $stage -le 6 ]; then
        gmm=exp/tri6b_cleaned
        dataset="test_clean dev_clean test_other dev_other"
        for set in $dataset
        do
                steps/nnet/decode.sh --nj 16 --cmd "$decode_cmd" \
		scoring_opts "--min-lmwt 10 --max-lmwt 30" \
                --config conf/decode_dnn.config --acwt $acwt \
                $gmm/graph_tgsmall \
                $data_fbk/$set $dir/decode_tgsmall_${set}

                steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                $data_fbk/$set $dir/decode_{tgsmall,tgmed}_${set}

                steps/lmrescore_const_arpa.sh \
		scoring_opts "--min-lmwt 10 --max-lmwt 30" \
                --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
                $data_fbk/$set $dir/decode_{tgsmall,tglarge}_${set}

                steps/lmrescore_const_arpa.sh \
		scoring_opts "--min-lmwt 10 --max-lmwt 30" \
                --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
                $data_fbk/$set $dir/decode_{tgsmall,fglarge}_${set}
        done
	for set in $dataset;
        do
        for lm in fglarge tglarge tgmed tgsmall;
        do
                grep WER $dir/decode_${lm}_${set}*/wer* | ./utils/best_wer.sh
        done
        done

fi

