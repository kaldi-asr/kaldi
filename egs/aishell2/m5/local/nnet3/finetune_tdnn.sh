# !/bin/bash
. ./path.sh
. ./cmd.sh

data_set=finetune

data_dir=data/${data_set}
feat_dir=data/${data_set}_hires
ali_dir=exp/${data_set}_ali

num_jobs_initial=1
num_jobs_final=1
num_epochs=5
initial_effective_lrate=0.0005
final_effective_lrate=0.00002
minibatch_size=1024

stage=1
nj=4

if [ $stage -le 1 ]; then
	# align new data(finetune set) with GMM, we probably replace GMM with NN later
	steps/make_mfcc.sh \
	    --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf \
			${data_dir} exp/make_mfcc/${data_set} mfcc
	steps/compute_cmvn_stats.sh ${data_dir} exp/make_mfcc/${data_set} mfcc || exit 1;

	utils/fix_data_dir.sh ${data_dir} || exit 1;
	steps/align_si.sh --cmd "$train_cmd" --nj ${nj} ${data_dir} data/lang exp/tri3 ${ali_dir}
	
	# extract fbank for AM finetuning
	utils/copy_data_dir.sh ${data_dir} ${feat_dir}
	rm -f ${feat_dir}/{cmvn.scp,feats.scp}
	utils/data/perturb_data_dir_volume.sh ${feat_dir} || exit 1;
	steps/make_fbank.sh \
	    --cmd "$train_cmd" --nj $nj --fbank-config conf/fbank.conf \
			${feat_dir} exp/make_fbank/${data_set} fbank
	steps/compute_cmvn_stats.sh ${feat_dir} exp/make_fbank/${data_set} fbank
fi

if [ $stage -le 2 ]; then
  dir=exp/nnet3/tdnn_finetune
	mkdir -p $dir
	
	local/nnet3/train_ft.sh \
		--num-jobs-initial $num_jobs_initial \
		--num-jobs-final $num_jobs_final \
		--num-epochs $num_epochs \
		--initial-effective-lrate $initial_effective_lrate \
		--final-effective-lrate $final_effective_lrate \
		--minibatch-size $minibatch_size \
		${feat_dir} data/lang ${ali_dir} exp/nnet3/tdnn_sp $dir
fi
