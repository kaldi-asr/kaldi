#!/bin/bash
# train a DNN on top of fMLLR features. 
# The training is done in 3 stages,
# 1) RBM pre-training:
#     unsupervised   train stack of RBMs, 
#     starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#     objective:  classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#     objective:  emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh
. ./path.sh
. ./conf/vars.sh
set -eu
set -o pipefail
. utils/parse_options.sh


steps/nnet/make_fmllr_feats.sh \
    --nj $nj --cmd "$train_cmd" \
    --transform-dir exp/tri5_semi_supervised/decode_test \
    data-fmllr-tri5_semi_supervised/test \
    data/test \
    exp/tri5_semi_supervised \
    data-fmllr-tri5_semi_supervised/test/log \
    data-fmllr-tri5_semi_supervised/test/data

steps/nnet/make_fmllr_feats.sh \
    --nj $nj --cmd "$train_cmd" \
    --transform-dir exp/tri5_semi_supervised_ali \
    data-fmllr-tri5_semi_supervised/train \
    data/train_semi_supervised \
    exp/tri5_semi_supervised \
	data-fmllr-tri5_semi_supervised/train/log \
	data-fmllr-tri5_semi_supervised/train/data

# split the data : 90% train 10% cross-validation (held-out)
utils/subset_data_dir_tr_cv.sh \
    data-fmllr-tri5_semi_supervised/train \
    data-fmllr-tri5_semi_supervised/train_tr90 \
    data-fmllr-tri5_semi_supervised/train_cv10

gmm=exp/tri5_semi_supervised
data_fmllr=data-fmllr-tri5_semi_supervised

# Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
dir=exp/dnn6_pretrain-dbn
$cuda_cmd \
    $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh \
    --hid-dim 1024 \
    --rbm-iter 20 \
    $data_fmllr/train \
    $dir

# Train the DNN optimizing per-frame cross-entropy.
dir=exp/dnn6_pretrain-dbn_dnn
ali=${gmm}_ali
feature_transform=exp/dnn6_pretrain-dbn/final.feature_transform
dbn=exp/dnn6_pretrain-dbn/6.dbn
# Train
$cuda_cmd \
    $dir/log/train_nnet.log \
    steps/nnet/train.sh \
    --feature-transform $feature_transform \
    --dbn $dbn \
    --hid-layers 0 \
    --learn-rate 0.008 \
    $data_fmllr/train_tr90 \
    $data_fmllr/train_cv10 \
    data/lang \
    $ali \
    $ali \
    $dir

# Decode (reuse HCLG graph)

steps/nnet/decode.sh \
    --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt 0.1 \
    $gmm/graph \
    $data_fmllr/test \
    $dir/decode_test

# prepare unigram lm
local/prepare_grammar_ug.sh

# make decoding fst graph
utils/mkgraph.sh     \
    --self-loop-scale 1.0     \
    data/lang_ug     \
    exp/dnn6_pretrain-dbn_dnn \
    exp/dnn6_pretrain-dbn_dnn/graph_ug

# decode test set with unigram lm
steps/nnet/decode.sh \
    --nj $nj \
    --cmd "$decode_cmd" \
    --config conf/decode_dnn.config \
    --acwt 0.1 \
    exp/dnn6_pretrain-dbn_dnn/graph_ug \
    data-fmllr-tri5_semi_supervised/test \
    exp/dnn6_pretrain-dbn_dnn/decode_ug_test || exit 1;

# generate  alignments:
steps/nnet/align.sh \
    --nj $nj \
    --cmd "$train_cmd" \
    data-fmllr-tri5/train \
    data/lang \
    exp/dnn6_pretrain-dbn_dnn \
    exp/dnn6_pretrain-dbn_dnn_ali || exit 1;

# generate denominator lattices
steps/nnet/make_denlats.sh \
    --nj $nj \
    --cmd "$decode_cmd" \
    --config conf/decode_dnn.config \
    --acwt 0.1 \
    data-fmllr-tri5/train \
    data/lang \
    exp/dnn6_pretrain-dbn_dnn \
    exp/dnn6_pretrain-dbn_dnn_denlats || exit 1;

# Sequence training using sMBR criterion Stochastic-GD with per-utterance updates.
# Re-train the DNN by 6 iterations of sMBR
steps/nnet/train_mpe.sh \
    --cmd "$cuda_cmd" \
    --num-iters 6 \
    --acwt 0.1 \
    --do-smbr true \
    data-fmllr-tri5/train \
    data/lang \
    exp/dnn6_pretrain-dbn_dnn \
    exp/dnn6_pretrain-dbn_dnn_ali \
    exp/dnn6_pretrain-dbn_dnn_denlats \
    exp/dnn6_pretrain-dbn_dnn_smbr || exit 1;

# make the decoding graph for dnn 
utils/mkgraph.sh \
    data/lang \
    exp/dnn6_pretrain-dbn_dnn_smbr \
    exp/dnn6_pretrain-dbn_dnn_smbr/dengraph || exit 1;

# decode test set
for ITER in 6 3 1; do
    steps/nnet/decode.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	--config conf/decode_dnn.config \
	--nnet exp/dnn6_pretrain-dbn_dnn_smbr/${ITER}.nnet \
	--acwt 0.1 \
	dnn6_pretrain-dbn_dnn_smbr/dengraph \
	data-fmllr-tri5/test \
	exp/dnn6_pretrain-dbn_dnn_smbr/decode_test_it${ITER}
done 

# p-norm nnet2 dnn
train_stage=-10
train_stage=-100

# train
steps/nnet2/train_pnorm_fast.sh \
    --stage $train_stage \
    --mix-up $dnn_mixup \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --cmd "$train_cmd" \
    "${dnn_gpu_parallel_opts[@]}" \
    data/train_semi_supervised \
    data/lang \
    exp/tri5_semi_supervised_ali \
    exp/tri6_nnet2 || exit 1

# decode
steps/nnet2/decode.sh \
    --cmd "$decode_cmd" \
    --nj $nj \
    --transform-dir exp/tri5_semi_supervised/decode_test \
    exp/tri5_semi_supervised/graph \
    data/test \
    exp/tri6_nnet2/decode_test || exit 1;

# nnet2 online
mkdir -p exp/nnet2_online

# train a universal background model to get started
steps/online/nnet2/train_diag_ubm.sh \
    --cmd "$train_cmd" \
    --nj 1 \
    --num-frames 300000 \
    data/train_semi_supervised \
    256 \
    exp/tri5_semi_supervised \
    exp/nnet2_online/diag_ubm || exit 1;

# train an iVector extractor
steps/online/nnet2/train_ivector_extractor.sh \
    --cmd "$train_cmd" \
    --nj $nj \
    data/train_semi_supervised \
    exp/nnet2_online/diag_ubm \
    exp/nnet2_online/extractor || exit 1;

# copy data
steps/online/nnet2/copy_data_dir.sh \
    --utts-per-spk-max 2 \
    data/train_semi_supervised \
    data/train_semi_supervised_max2 || exit 1;

# extract iVector from train data
steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" \
    --nj $nj \
    data/train_semi_supervised_max2 \
    exp/nnet2_online/extractor \
    exp/nnet2_online/ivectors_train_semi_supervised || exit 1;

# extract iVector from test data
steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" \
    --nj $nj \
    data/test \
    exp/nnet2_online/extractor \
    exp/nnet2_online/ivectors_test || exit 1;

# train
train_stage=-10
exit_train_stage=-100
num_threads=1
minibatch_size=512
parallel_opts="--gpu 1" 

steps/nnet2/train_multisplice_accel2.sh \
    --stage $train_stage \
    --exit-stage $exit_train_stage \
    --num-epochs 8 \
    --num-jobs-initial 2 \
    --num-jobs-final $nj \
    --num-hidden-layers 4 \
    --splice-indexes "layer0/-1:0:1 layer1/-2:1 layer2/-4:2" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_semi_supervised \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --initial-effective-lrate 0.005 \
    --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 250 \
    --mix-up 12000 \
    data/train_semi_supervised \
    data/lang \
    exp/tri5_semi_supervised_ali \
    exp/nnet2_online/nnet_ms_a  || exit 1;
