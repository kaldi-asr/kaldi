#!/bin/bash
set -e
set -o pipefail
set u
. ./cmd.sh
. ./path.sh
nj=10
stage=1
. ./utils/parse_options.sh
# data locations
gp_train_data=/mnt/corpora/Globalphone/gp/FRF_ASR003/wav
yaounde_train_data=/mnt/corpora/Yaounde/read/wavs/16000
ca_test_data=/mnt/corpora/central_accord/test
answers_data=/mnt/corpora/Yaounde/answers/16000
sri_gabon_data=/mnt/corpora/sri_gabon

# prepare transcriptions
if [ $stage -le 1 ]; then
    tmp_dir=data/local/tmp
    # delete the existing tmp directory
    if [ -d $tmp_dir ]; then
	rm -Rf $tmp_dir
    fi

    mkdir -p $tmp_dir

    # delete the existing prompts directory
    if [ -d data/prompts ]; then
	rm -Rf data/prompts
    fi

    # delete the existing wavs directory
    if [ -d data/wavs ]; then
	rm -Rf data/wavs
    fi
    
    # prep gp data for training
    local/gp_get_data.sh \
	$gp_train_data \
	$tmp_dir

    # yaounde train data prep
    local/yaounde_get_data.sh \
	$yaounde_train_data \
	$tmp_dir

    # prepare the Central Accord test data
    local/central_accord_get_data.sh \
	$ca_test_data \
	$tmp_dir

    # prepare the yaounde answers data
    local/yaounde_answers_get_data.sh \
	$answers_data \
	$tmp_dir

    mkdir -p data/train
    
    # sort the gp text
    sort \
	$tmp_dir/gp_utt2text_unsorted.txt > \
	$tmp_dir/train_text

    # append the yaounde data to the text
    sort \
	$tmp_dir/yaounde_utt2text_unsorted.txt >> \
	$tmp_dir/train_text

    # sort the 2 data texts
    sort \
	$tmp_dir/train_text > \
	data/train/text

    # sort the wav files
    sort \
	$tmp_dir/gp_wav_unsorted.scp > \
	$tmp_dir/train_wav.scp

    sort \
	$tmp_dir/yaounde_wav_unsorted.scp >> \
	$tmp_dir/train_wav.scp

    sort \
	$tmp_dir/train_wav.scp > \
	data/train/wav.scp

    # sort the speaker to utterance mappings
    sort \
	$tmp_dir/gp_spk2utt_unsorted.txt > \
	$tmp_dir/train_spk2utt

    sort \
	$tmp_dir/yaounde_spk2utt_unsorted.txt >> \
	$tmp_dir/train_spk2utt

    sort \
	$tmp_dir/train_spk2utt > \
	data/train/spk2utt

    # sort the utterance to speaker mappings
    sort \
	$tmp_dir/gp_utt2spk_unsorted.txt > \
	$tmp_dir/train_utt2spk

    sort \
	$tmp_dir/yaounde_utt2spk_unsorted.txt >> \
	$tmp_dir/train_utt2spk

    sort \
	$tmp_dir/train_utt2spk > \
	data/train/utt2spk

    # prepare dictionaries and lexicons
    mkdir -p data/local/dict
    cp local/src/extra_questions.txt local/src/lexicon.txt \
       local/src/lexiconp.txt local/src/nonsilence_phones.txt \
       local/src/optional_silence.txt local/src/silence_phones.txt \
       data/local/dict
    utils/prepare_lang.sh \
	--position-dependent-phones false \
	--sil-prob 0.5 \
	data/local/dict "<SPOKEN_NOISE>" \
	data/local/lang_tmp \
	data/lang || exit 1;

    # extract front end features
    plp_dir=plp
    mkdir -p exp

    for part in train test answers; do
	if [ -e data/$part/cmvn.scp ]; then
	    rm data/$part/cmvn.scp
	fi
	steps/make_plp_pitch.sh \
	    --cmd "$train_cmd" --nj $train_nj \
	    data/$part \
	    exp/make_plp_pitch/$part \
	    $plp_dir || exit 1;
	utils/fix_data_dir.sh \
	    data/$part || exit 1;
	steps/compute_cmvn_stats.sh \
	    data/$part \
	    exp/make_plp/$part \
	    $plp_dir || exit 1;
	utils/fix_data_dir.sh \
	    data/$part || exit 1;
    done
fi

# train monophones and 4 passes of triphone training
if [ $stage -le 2 ]; then
    if [ ! -f data/train_sub3/.done ]; then
	echo "Subsetting monophone training data in data/train_sub[123] on" `date`
	numutt=`cat data/train/feats.scp | wc -l`;
	utils/subset_data_dir.sh \
	    data/train  \
	    5000 \
	    data/train_sub1
	if [ $numutt -gt 10000 ] ; then
	    utils/subset_data_dir.sh \
		data/train \
		10000 \
		data/train_sub2
	else
	    (cd data; ln -s train train_sub2 )
	fi
	if [ $numutt -gt 20000 ] ; then
	    utils/subset_data_dir.sh \
		data/train \
		20000 \
		data/train_sub3
	else
	    (cd data; ln -s train train_sub3 )
	fi

	touch data/train_sub3/.done
    fi

    # train context independent monophones
    steps/train_mono.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_sub1 \
	data/lang \
	exp/mono || exit 1;

    # align with monophones
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_sub2 \
	data/lang \
	exp/mono \
	exp/mono_ali_sub2

    # train tri1
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
    --cmd "$train_cmd" \
    $numLeavesTri1 \
    $numGaussTri1 \
    data/train_sub2 \
    data/lang \
    exp/mono_ali_sub2 \
    exp/tri1 || exit 1;

    # align with tri1
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_sub3 \
	data/lang \
	exp/tri1 \
	exp/tri1_ali_sub3 || exit 1;

    # train tri2
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri2 \
	$numGaussTri2 \
	data/train_sub3 \
	data/lang \
	exp/tri1_ali_sub3 \
	exp/tri2 || exit 1;

    # align with tri2
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri2 \
	exp/tri2_ali || exit 1;

    # train tri3
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri3 \
	$numGaussTri3 \
	data/train \
	data/lang \
	exp/tri2_ali \
	exp/tri3 || exit 1;

    # align with tri3
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri3 \
	exp/tri3_ali || exit 1;

    # train tri4 with lda and mllt
    steps/train_lda_mllt.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesMLLT \
	$numGaussMLLT \
	data/train \
	data/lang \
	exp/tri3_ali \
	exp/tri4 || exit 1;

    # align with tri4
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri4 \
	exp/tri4_ali || exit 1;

    # speaker adaptive training tri5
    steps/train_sat.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesSAT \
	$numGaussSAT \
	data/train \
	data/lang \
	exp/tri4_ali \
	exp/tri5 || exit 1;
fi

# start SGMM training
if [ $stage -le 3 ]; then
    echo "Starting exp/tri5_ali on" `date`
    # do fmmlr alignment with tri5
    steps/align_fmllr.sh \
	--boost-silence 0.125 \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri5 \
	exp/tri5_ali || exit 1;

    # train a universal background model ubm5
    steps/train_ubm.sh \
	--cmd "$train_cmd" \
	$numGaussUBM \
	data/train \
	data/lang \
	exp/tri5_ali \
	exp/ubm5 || exit 1;

    # train sgmm
    steps/train_sgmm2.sh \
	--cmd "$train_cmd" \
	$numLeavesSGMM \
	$numGaussSGMM \
	data/train \
	data/lang \
	exp/tri5_ali \
	exp/ubm5/final.ubm \
	exp/sgmm5 || exit 1;

    #start discriminative SGMM training
    steps/align_sgmm2.sh \
	--nj $train_nj \
	--cmd "$train_cmd" \
	--transform-dir exp/tri5_ali \
	--use-graphs true \
	--use-gselect true \
	data/train \
	data/lang \
	exp/sgmm5 \
	exp/sgmm5_ali || exit 1;

    # make denoninator lattices
    steps/make_denlats_sgmm2.sh \
	--nj $train_nj \
	--sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
	--beam 10.0 \
	--lattice-beam 6 \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_ali \
	data/train \
	data/lang \
	exp/sgmm5_ali \
	exp/sgmm5_denlats || exit 1;

    # train with boosted mmi
    steps/train_mmi_sgmm2.sh \
	--cmd "$train_cmd" \
	"${sgmm_mmi_extra_opts[@]}" \
	--drop-frames true \
	--transform-dir exp/tri5_ali \
	--boost 0.1 \
	data/train \
	data/lang \
	exp/sgmm5_ali \
	exp/sgmm5_denlats \
	exp/sgmm5_mmi_b0.1 || exit 1;
fi

if [ $stage -le 4 ]; then
    # make fst graph for decoding with tri5 models
    utils/mkgraph.sh \
	data/lang  \
	exp/tri5 \
	exp/tri5/graph || exit 1;

    # ditto for sgmm models
    utils/mkgraph.sh \
	data/lang \
	exp/sgmm5 \
	exp/sgmm5/graph || exit 1;

    # decode test data with tri5 models to get transforms
    steps/decode_fmllr.sh \
	--nj $nj  \
	exp/tri5/graph  \
	data/test \
	exp/tri5/decode_test || exit 1;

    #decode answers data with tri5 models to get transforms
    steps/decode_fmllr.sh \
	--nj $nj --cmd "$decode_cmd" \
	exp/tri5/graph \
	data/answers \
	exp/tri5/decode_answers || exit 1;

    # decode test with sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5/decode_test \
	exp/sgmm5/graph \
	data/test \
	exp/sgmm5/decode_test || exit 1;

    # decode answers with sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5/decode_answers \
	exp/sgmm5/graph \
	data/answers \
	exp/sgmm5/decode_answers || exit 1;

    # decode test with sgmm and mllr
    steps/decode_sgmm2.sh \
	--use-fmllr true \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5/decode_test \
	exp/sgmm5/graph \
	data/test \
	exp/sgmm5/decode_fmllr_test || exit 1;

    # decode answers with sgmm and mllr
    steps/decode_sgmm2.sh \
	--use-fmllr true \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5/decode_answers \
	exp/sgmm5/graph \
	data/answers \
	exp/sgmm5/decode_fmllr_answers || exit 1;

    # decode test with boosted mmi trained sgmm and lattice rescoring
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5/decode_test \
	    data/lang \
	    data/test \
	    exp/sgmm5/decode_test \
	    exp/sgmm5_mmi_b0.1/decode_test_it$iter &
    done  

    # ditto  for answers 
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5/decode_answers \
	    data/lang \
	    data/answers \
	    exp/sgmm5/decode_answers \
	    exp/sgmm5_mmi_b0.1/decode_answers_it$iter &
    done  
fi

if [ $stage -le 5 ]; then
    # prepare the data/train directory for semi supervised training

    tmp_dir=data/local/tmp
    data_dir=data/train_semi_supervised
    if [ -d $data_dir ]; then
	rm -Rf $data_dir
    fi

    mkdir -p $data_dir

    # get the semi supervised transcripts
    local/yaounde_answers_per_utt2text.pl \
	exp/sgmm5_mmi_b0.1/decode_answers_it4/scoring_kaldi/wer_details/per_utt > \
	local/src/yaounde_answers_best_wer_transcripts.txt

    yaounde_answers_supervision=local/src/yaounde_answers_best_wer_transcripts.txt

    # point to the best central accord transcripts we have 
    sri_gabon_read_supervision=local/src/sri_gabon_read_best_wer_transcripts.txt

    # make the new spk2utt file with both the old train and answers
    cat \
	data/train/spk2utt \
	data/answers/spk2utt > \
	$tmp_dir/spk2utt.unsorted

    sort \
	$tmp_dir/spk2utt.unsorted > \
	$data_dir/spk2utt

    # ditto for utt2spk
    cat \
	data/train/utt2spk \
	data/answers/utt2spk > \
	$tmp_dir/utt2spk.unsorted

    sort \
	$tmp_dir/utt2spk.unsorted > \
	$data_dir/utt2spk

    # ditto for the wav.scp file
    cat \
	data/train/wav.scp \
	data/answers/wav.scp > \
	$tmp_dir/wav.scp.unsorted

    sort \
	$tmp_dir/wav.scp.unsorted > \
	$data_dir/wav.scp

    # ditto for the text file but  concatenate with the output from  recognizer
    if [ -e $tmp_dir/text ]; then
	rm $tmp_dir/text
    fi

    # concatenate the old text with the answers hyps
    cat \
	data/train/text \
	$yaounde_answers_supervision > \
	$tmp_dir/text

    sort \
	$tmp_dir/text > \
	$data_dir/text

    # use the sgmm transcripts as reference for testing
    cp \
	data/answers/text \
	data/answers/text.1

    cp \
	$yaounde_answers_supervision \
	data/answers/text

    # prepare the central accord read data
    local/sri_gabon_read_get_data.sh \
	$sri_gabon_data \
	$tmp_dir \
	$sri_gabon_read_supervision

    for part in train_semi_supervised sri_gabon_read ; do
	if [ -e data/$part/cmvn.scp ]; then
	    rm data/$part/cmvn.scp
	fi
	steps/make_plp_pitch.sh \
	    --cmd "$train_cmd" --nj $train_nj \
	    data/$part \
	    exp/make_plp_pitch/$part \
	    $plp_dir || exit 1;
	utils/fix_data_dir.sh \
	    data/$part || exit 1;
	steps/compute_cmvn_stats.sh \
	    data/$part \
	    exp/make_plp/$part \
	    $plp_dir || exit 1;
	utils/fix_data_dir.sh \
	    data/$part || exit 1;
    done
fi

# monophone and 4 passes of triphone training on semi supervised data
if [ $stage -le 6 ];  then
    if [ ! -f data/train_semi_supervised_sub3/.done ]; then
	echo "Subsetting  data in data/train_semi_supervised_sub[123] on" `date`
	numutt=`cat data/train_semi_supervised/feats.scp | wc -l`;
	utils/subset_data_dir.sh \
	    data/train_semi_supervised \
	    5000 \
	    data/train_semi_supervised_sub1
	if [ $numutt -gt 10000 ] ; then
	    utils/subset_data_dir.sh \
		data/train_semi_supervised \
		10000 \
		data/train_semi_supervised_sub2
	else
	    (cd data; ln -s train train_semi_supervised_sub2 )
	fi
	if [ $numutt -gt 20000 ] ; then
	    utils/subset_data_dir.sh \
		data/train_semi_supervised \
		20000 \
		data/train_semi_supervised_sub3
	else
	    (cd data; ln -s train_semi_supervised train_semi_supervised_sub3 )
	fi

	touch data/train_semi_supervised_sub3/.done
    fi

    # train monophones on semi supervised data
    steps/train_mono.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_sub1 \
	data/lang \
	exp/mono_semi_supervised || exit 1;

    # align with semi supervised monophones
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_sub2 \
	data/lang \
	exp/mono_semi_supervised \
	exp/mono_semi_supervised_sub2_ali || exit 1;

    # train tri1 on semi supervised data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri1 \
	$numGaussTri1 \
	data/train_semi_supervised_sub2 \
	data/lang \
	exp/mono_semi_supervised_sub2_ali \
	exp/tri1_semi_supervised || exit 1;

    # align with semi supervised tri1
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_sub3 \
	data/lang \
	exp/tri1_semi_supervised \
	exp/tri1_semi_supervised_sub3_ali || exit 1;

    # train tri2 on semi supervised data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri2 \
	$numGaussTri2 \
	data/train_semi_supervised_sub3 \
	data/lang \
	exp/tri1_semi_supervised_sub3_ali \
	exp/tri2_semi_supervised || exit 1;

    # align with semi supervised tri2
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised \
	data/lang \
	exp/tri2_semi_supervised \
	exp/tri2_semi_supervised_ali || exit 1;

    # train tri3 on semi supervised data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri3 \
	$numGaussTri3 \
	data/train_semi_supervised \
	data/lang \
	exp/tri2_semi_supervised_ali \
	exp/tri3_semi_supervised || exit 1;

    # align with tri3 on semi supervised data
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised \
	data/lang \
	exp/tri3_semi_supervised \
	exp/tri3_semi_supervised_ali || exit 1;

    # train tri4 with lda and mllt on semi supervised data
    steps/train_lda_mllt.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesMLLT \
	$numGaussMLLT \
	data/train_semi_supervised \
	data/lang \
	exp/tri3_semi_supervised_ali \
	exp/tri4_semi_supervised || exit 1;

    # align with semi supervised trained tri4
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised \
	data/lang \
	exp/tri4_semi_supervised \
	exp/tri4_semi_supervised_ali || exit 1;
fi

if [ $stage -le 7 ]; then
    # speaker adaptive training tri5 on semi supervised
    steps/train_sat.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesSAT \
	$numGaussSAT \
	data/train_semi_supervised \
	data/lang \
	exp/tri4_semi_supervised_ali \
	exp/tri5_semi_supervised || exit 1;

    # SGMM training on semi supervised data starts here
    echo "Starting exp/tri5_semi_supervised_ali on" `date`
    # do fmmlr alignment with semi supervised trained tri5
    steps/align_fmllr.sh \
	--boost-silence 0.125 \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised \
	data/lang \
	exp/tri5_semi_supervised \
	exp/tri5_semi_supervised_ali || exit 1;

    # train a universal background model ubm5 on semi supervised data
    steps/train_ubm.sh \
	--cmd "$train_cmd" \
	$numGaussUBM \
	data/train_semi_supervised \
	data/lang \
	exp/tri5_semi_supervised_ali \
	exp/ubm5_semi_supervised || exit 1;

    # train sgmm on semi supervised data
    steps/train_sgmm2.sh \
	--cmd "$train_cmd" \
	$numLeavesSGMM \
	$numGaussSGMM \
	data/train_semi_supervised \
	data/lang \
	exp/tri5_semi_supervised_ali \
	exp/ubm5_semi_supervised/final.ubm \
	exp/sgmm5_semi_supervised || exit 1;

    #start discriminative SGMM training on semi supervised data
    steps/align_sgmm2.sh \
	--nj $train_nj \
	--cmd "$train_cmd" \
	--transform-dir exp/tri5_semi_supervised_ali \
	--use-graphs true \
	--use-gselect true \
	data/train_semi_supervised \
	data/lang \
	exp/sgmm5_semi_supervised \
	exp/sgmm5_semi_supervised_ali || exit 1;

    # make denoninator lattices with semi supervised data
    steps/make_denlats_sgmm2.sh \
	--nj $train_nj \
	--sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
	--beam 10.0 \
	--lattice-beam 6 \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_ali \
	data/train_semi_supervised \
	data/lang \
	exp/sgmm5_semi_supervised_ali \
	exp/sgmm5_semi_supervised_denlats || exit 1;

    # train with boosted mmi on semi supervised data
    steps/train_mmi_sgmm2.sh \
	--cmd "$train_cmd" \
	"${sgmm_mmi_extra_opts[@]}" \
	--drop-frames true \
	--transform-dir exp/tri5_semi_supervised_ali \
	--boost 0.1 \
	data/train_semi_supervised \
	data/lang \
	exp/sgmm5_semi_supervised_ali \
	exp/sgmm5_semi_supervised_denlats \
	exp/sgmm5_semi_supervised_mmi_b0.1 || exit 1;
fi

if [ $stage -le 8 ]; then
    # make decoding fst for semi supervised trained tri5 models
    utils/mkgraph.sh \
	data/lang  \
	exp/tri5_semi_supervised \
	exp/tri5_semi_supervised/graph || exit 1;

    # decode test data with semi supervised trained tri5 models to get transforms
    steps/decode_fmllr.sh \
	--nj $nj  \
	exp/tri5_semi_supervised/graph  \
	data/test \
	exp/tri5_semi_supervised/decode_test || exit 1;

    #decode answers data with semi supervised trained tri5 models to get transforms
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised/graph \
	data/answers \
	exp/tri5_semi_supervised/decode_answers || exit 1;

    #decode sri_gabon read data with semi supervised trained tri5 models 
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised/graph \
	data/sri_gabon_read \
	exp/tri5_semi_supervised/decode_sri_gabon_read || exit 1;

    # make the  graph for decoding with semi supervised trained sgmm
    utils/mkgraph.sh \
	data/lang \
	exp/sgmm5_semi_supervised \
	exp/sgmm5_semi_supervised/graph || exit 1;

# decode test with semi supervised trained sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised/decode_test \
	exp/sgmm5_semi_supervised/graph \
	data/test \
	exp/sgmm5_semi_supervised/decode_test || exit 1;

    # decode answers with semi supervised trained sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised/decode_answers \
	exp/sgmm5_semi_supervised/graph \
	data/answers \
	exp/sgmm5_semi_supervised/decode_answers || exit 1;

    # decode sri_gabon read with semi supervised trained sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised/decode_sri_gabon_read \
	exp/sgmm5_semi_supervised/graph \
	data/sri_gabon_read \
	exp/sgmm5_semi_supervised/decode_sri_gabon_read || exit 1;

    # decode test with boosted mmi semi supervised trained sgmm 
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised/decode_test \
	    data/lang \
	    data/test \
	    exp/sgmm5_semi_supervised/decode_test \
	    exp/sgmm5_semi_supervised_mmi_b0.1/decode_test_it$iter &
    done  

    # ditto for answers 
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised/decode_answers \
	    data/lang \
	    data/answers \
	    exp/sgmm5_semi_supervised/decode_answers \
	    exp/sgmm5_semi_supervised_mmi_b0.1/decode_answers_it$iter &
    done  

    # ditto for central accord read 
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised/decode_sri_gabon_read \
	    data/lang \
	    data/sri_gabon_read \
	    exp/sgmm5_semi_supervised/decode_sri_gabon_read \
	    exp/sgmm5_semi_supervised_mmi_b0.1/decode_sri_gabon_read_it$iter &
    done  
fi

if [ $stage -le 9 ]; then
    #  set the path to the best transcripts from the previous stage
    local/sri_gabon_read_per_utt2text.pl \
	exp/sgmm5_semi_supervised_mmi_b0.1/decode_sri_gabon_read_it4/scoring_kaldi/wer_details/per_utt > \
	local/src/sri_gabon_read_best_wer_transcripts.txt

    sri_gabon_read_supervision=local/src/sri_gabon_read_best_wer_transcripts.txt
    answers_supervision=local/src/yaounde_answers_best_wer_transcripts.txt

    # directory for temporary working files
    tmp_dir=data/local/tmp
    rm -Rf $tmp_dir/sri_gabon_read_*
    # directory for semi supervised 2 training data
    data_dir=data/train_semi_supervised_2

    if [ -d $data_dir ]; then
	rm -Rf $data_dir
    fi

    mkdir -p $data_dir

    # concatenate spk2utt file from previous stage to sri_gabon_read spk2utt
    cat \
	data/train_semi_supervised/spk2utt \
	data/sri_gabon_read/spk2utt > \
	$tmp_dir/spk2utt.unsorted
    sort \
	$tmp_dir/spk2utt.unsorted > \
	$data_dir/spk2utt

    # ditto for the utt2spk file
    cat \
	data/train_semi_supervised/utt2spk \
	data/sri_gabon_read/utt2spk > \
	$tmp_dir/utt2spk.unsorted
    sort \
	$tmp_dir/utt2spk.unsorted > \
	$data_dir/utt2spk

    # ditto for the wav.scp file
    cat \
	data/train_semi_supervised/wav.scp \
	data/sri_gabon_read/wav.scp > \
	$tmp_dir/wav.scp.unsorted

    sort \
	$tmp_dir/wav.scp.unsorted > \
	$data_dir/wav.scp

    #  the sri_gabon read text  file
    # here is where we use the transcripts from the previous stage 
    if [ -e $tmp_dir/text ]; then
	rm $tmp_dir/text
    fi

    cat \
	data/train_semi_supervised/text \
	$sri_gabon_read_supervision > \
	$tmp_dir/text

    sort  \
	$tmp_dir/text > \
	$data_dir/text

    # use the best sgmm semi supervised transcripts as reference for testing
    cp \
	data/answers/text \
	data/answers/text.2

    cp \
	$answers_supervision \
	data/answers/text

    # ditto for the best sri_gabon read transcripts
    cp \
	data/sri_gabon_read/text \
	data/sri_gabon_read/text.1

    cp \
	$sri_gabon_read_supervision \
	data/sri_gabon_read/text

    # extract plp pitch features
    for part in train_semi_supervised_2 ; do
	if [ -e data/$part/cmvn.scp ]; then
	    rm data/$part/cmvn.scp
	fi

	steps/make_plp_pitch.sh \
	    --cmd "$train_cmd" \
	    --nj $train_nj \
	    data/$part \
	    exp/make_plp_pitch/$part \
	    $plp_dir || exit 1;

	utils/fix_data_dir.sh \
	    data/$part || exit 1;

	steps/compute_cmvn_stats.sh \
	    data/$part \
	    exp/make_plp/$part \
	    $plp_dir || exit 1;

	utils/fix_data_dir.sh \
	    data/$part || exit 1;

    done
fi

# train monophones and 4 passes of triphones with semi supervised 2 data
if [ $stage -le 10 ]; then
    if [ ! -f data/train_semi_supervised_2_sub3/.done ]; then
	echo "Subsetting in data/train_semi_supervised_2_sub[123] on" `date`
	numutt=`cat data/train_semi_supervised_2/feats.scp | wc -l`;
	utils/subset_data_dir.sh \
	    data/train_semi_supervised_2 \
	    5000 \
	    data/train_semi_supervised_2_sub1
	if [ $numutt -gt 10000 ] ; then
	    utils/subset_data_dir.sh \
		data/train_semi_supervised_2 \
		10000 \
		data/train_semi_supervised_2_sub2
	else
	    (cd data; ln -s train train_semi_supervised_2_sub2 )
	fi
	if [ $numutt -gt 20000 ] ; then
	    utils/subset_data_dir.sh \
		data/train_semi_supervised_2 \
		20000 \
		data/train_semi_supervised_2_sub3
	else
	    (cd data; ln -s train_semi_supervised_2 train_semi_supervised_2_sub3 )
	fi

	touch data/train_semi_supervised_2_sub3/.done
    fi

    # train  monophones with semi supervised 2 data
    steps/train_mono.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2_sub1 \
	data/lang \
	exp/mono_semi_supervised_2 || exit 1;

    # align with semi supervised 2 trained monophones
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2_sub2 \
	data/lang \
	exp/mono_semi_supervised_2 \
	exp/mono_semi_supervised_2_sub2_ali

    # train tri1 on semi supervised data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri1 \
	$numGaussTri1 \
	data/train_semi_supervised_2_sub2 \
	data/lang \
	exp/mono_semi_supervised_2_sub2_ali \
	exp/tri1_semi_supervised_2 || exit 1;

    # align with semi supervised trained models tri1
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2_sub3 \
	data/lang \
	exp/tri1_semi_supervised_2 \
	exp/tri1_semi_supervised_2_sub3_ali || exit 1;

    # train tri2 on semi supervised data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri2 \
	$numGaussTri2 \
	data/train_semi_supervised_2_sub3 \
	data/lang \
	exp/tri1_semi_supervised_2_sub3_ali \
	exp/tri2_semi_supervised_2 || exit 1;

    # align with semi supervised 2 trained models tri2
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri2_semi_supervised_2 \
	exp/tri2_semi_supervised_2_ali || exit 1;

    # train semi supervised tri3 models
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri3 \
	$numGaussTri3 \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri2_semi_supervised_2_ali \
	exp/tri3_semi_supervised_2 || exit 1;

    # align with semi supervised 2 trained tri3 models
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri3_semi_supervised_2 \
	exp/tri3_semi_supervised_2_ali || exit 1;

    # train tri4 with lda and mllt on semi supervised 2 data
    steps/train_lda_mllt.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesMLLT \
	$numGaussMLLT \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri3_semi_supervised_2_ali \
	exp/tri4_semi_supervised_2 || exit 1;

    # align with semi supervised 2 tri4 models
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri4_semi_supervised_2 \
	exp/tri4_semi_supervised_2_ali || exit 1;
fi

if [ $stage -le 11 ]; then
    # train speaker adaptive tri5 models on semi supervised 2 data 
    steps/train_sat.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesSAT \
	$numGaussSAT \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri4_semi_supervised_2_ali \
	exp/tri5_semi_supervised_2 || exit 1;

    # start SGMM training on semi supervised 2 data
    echo "Starting exp/tri5_semi_supervised_2_ali on" `date`
    # do fmmlr alignment with tri5 on semi supervised 2 data
    steps/align_fmllr.sh \
	--boost-silence 0.125 \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri5_semi_supervised_2 \
	exp/tri5_semi_supervised_2_ali || exit 1;

    # train a universal background model ubm5 on semi supervised 2 data
    steps/train_ubm.sh \
	--cmd "$train_cmd" \
	$numGaussUBM \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri5_semi_supervised_2_ali \
	exp/ubm5_semi_supervised_2 || exit 1;

    # train sgmm
    steps/train_sgmm2.sh \
	--cmd "$train_cmd" \
	$numLeavesSGMM \
	$numGaussSGMM \
	data/train_semi_supervised_2 \
	data/lang \
	exp/tri5_semi_supervised_2_ali \
	exp/ubm5_semi_supervised_2/final.ubm \
	exp/sgmm5_semi_supervised_2 || exit 1;

    #start discriminative SGMM training with alignment on semi supervised 2 data
    steps/align_sgmm2.sh \
	--nj $train_nj \
	--cmd "$train_cmd" \
	--transform-dir exp/tri5_semi_supervised_2_ali \
	--use-graphs true \
	--use-gselect true \
	data/train_semi_supervised_2 \
	data/lang \
	exp/sgmm5_semi_supervised_2 \
	exp/sgmm5_semi_supervised_2_ali || exit 1;

    # make denominator lattices on semi supervised 2 data
    steps/make_denlats_sgmm2.sh \
	--nj $train_nj \
	--sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
	--beam 10.0 \
	--lattice-beam 6 \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_2_ali \
	data/train_semi_supervised_2 \
	data/lang \
	exp/sgmm5_semi_supervised_2_ali \
	exp/sgmm5_semi_supervised_2_denlats || exit 1;

    # train with boosted mmi on semi supervised 2 data
    steps/train_mmi_sgmm2.sh \
	--cmd "$train_cmd" \
	"${sgmm_mmi_extra_opts[@]}" \
	--drop-frames true \
	--transform-dir exp/tri5_semi_supervised_2_ali \
	--boost 0.1 \
	data/train_semi_supervised_2 \
	data/lang \
	exp/sgmm5_semi_supervised_2_ali \
	exp/sgmm5_semi_supervised_2_denlats \
	exp/sgmm5_semi_supervised_2_mmi_b0.1 || exit 1;
fi

if [ $stage -le 12 ]; then
    # make fst for tri5 models
    utils/mkgraph.sh \
	data/lang  \
	exp/tri5_semi_supervised_2 \
	exp/tri5_semi_supervised_2/graph || exit 1;

    # decode test data with semi supervised 2 trained tri5 models
    # to get transforms
    steps/decode_fmllr.sh \
	--nj $nj  \
	exp/tri5_semi_supervised_2/graph  \
	data/test \
	exp/tri5_semi_supervised_2/decode_test || exit 1;

    # ditto for answers
    steps/decode_fmllr.sh \
	--nj $nj --cmd "$decode_cmd" \
	exp/tri5_semi_supervised_2/graph \
	data/answers \
	exp/tri5_semi_supervised_2/decode_answers || exit 1;

    #ditto  for sri_gabon_read data 
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised_2/graph \
	data/sri_gabon_read \
	exp/tri5_semi_supervised_2/decode_sri_gabon_read || exit 1;

    # prepare sri_gabon_conv data
    # start with what ever we have available
    sri_gabon_conv_supervision=local/src/sri_gabon_conv_best_wer_transcripts.txt
    sri_gabon_read_supervision=local/src/sri_gabon_read_best_wer_transcripts.txt
    answers_supervision=local/src/yaounde_answers_best_wer_transcripts.txt

    # directory for temporary working files
    tmp_dir=data/local/tmp
    rm -Rf $tmp_dir/sri_gabon_conv_*

    # prepare the sri_gabon_conv data
    local/sri_gabon_conv_get_data.sh \
	$sri_gabon_data \
	$tmp_dir \
	$sri_gabon_conv_supervision

    # extract plp pitch features for sri_gabon_conv data

    for part in sri_gabon_conv ; do
	if [ -e data/$part/cmvn.scp ]; then
	rm data/$part/cmvn.scp
	fi

	steps/make_plp_pitch.sh \
	    --cmd "$train_cmd" \
	    --nj $train_nj \
	    data/$part \
	    exp/make_plp_pitch/$part \
	    $plp_dir || exit 1;

	utils/fix_data_dir.sh \
	    data/$part || exit 1;
	steps/compute_cmvn_stats.sh \
	    data/$part \
	    exp/make_plp/$part \
	    $plp_dir || exit 1;

	utils/fix_data_dir.sh \
	    data/$part || exit 1;
    done

    #decode sri_gabon_conv data with tri5 models to get transforms
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised_2/graph \
	data/sri_gabon_conv \
	exp/tri5_semi_supervised_2/decode_sri_gabon_conv || exit 1;

    # make the decoding graph 
    utils/mkgraph.sh \
	data/lang \
	exp/sgmm5_semi_supervised_2 \
	exp/sgmm5_semi_supervised_2/graph || exit 1;

    # decode test with semi supervised 2 trained sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_2/decode_test \
	exp/sgmm5_semi_supervised_2/graph \
	data/test \
	exp/sgmm5_semi_supervised_2/decode_test || exit 1;

    # ditto for answers
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_2/decode_answers \
	exp/sgmm5_semi_supervised_2/graph \
	data/answers \
	exp/sgmm5_semi_supervised_2/decode_answers || exit 1;

    # decode sri_gabon read with semi supervised 2 trained sgmm models
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_2/decode_sri_gabon_read \
	exp/sgmm5_semi_supervised_2/graph \
	data/sri_gabon_read \
	exp/sgmm5_semi_supervised_2/decode_sri_gabon_read || exit 1;

    # decode test with semi supervised 2 trained sgmm models
    # and lattice rescoring
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised_2/decode_test \
	    data/lang \
	    data/test \
	    exp/sgmm5_semi_supervised_2/decode_test \
	    exp/sgmm5_semi_supervised_2_mmi_b0.1/decode_test_it$iter &
    done  

    # ditto for answers
    for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh \
	--cmd "$decode_cmd" \
	--iter $iter \
	--transform-dir exp/tri5_semi_supervised_2/decode_answers \
	data/lang \
	data/answers \
	exp/sgmm5_semi_supervised_2/decode_answers \
	exp/sgmm5_semi_supervised_2_mmi_b0.1/decode_answers_it$iter &
    done  

    # ditto for central accord read data
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	--cmd "$decode_cmd" \
	--iter $iter \
	--transform-dir exp/tri5_semi_supervised_2/decode_sri_gabon_read \
	data/lang \
	data/sri_gabon_read \
	exp/sgmm5_semi_supervised_2/decode_sri_gabon_read \
	exp/sgmm5_semi_supervised_2_mmi_b0.1/decode_sri_gabon_read_it$iter &
    done  

    # decode sri_gabon_conv with sgmm 
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_2/decode_sri_gabon_conv \
	exp/sgmm5_semi_supervised_2/graph \
	data/sri_gabon_conv \
	exp/sgmm5_semi_supervised_2/decode_sri_gabon_conv || exit 1;

    # decode sri_gabon_conv with boosted mmi sgmm 
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised_2/decode_sri_gabon_conv \
	    data/lang \
	    data/sri_gabon_conv \
	    exp/sgmm5_semi_supervised_2/decode_sri_gabon_conv \
	    exp/sgmm5_semi_supervised_2_mmi_b0.1/decode_sri_gabon_conv_it$iter &
    done  
fi

if [ $stage -le 13 ]; then
    local/sri_gabon_conv_per_utt2text.pl \
	exp/sgmm5_semi_supervised_2_mmi_b0.1/decode_sri_gabon_conv_it4 > \
	local/src/sri_gabon_conv_best_wer_transcripts.txt

    #  set the path to the best transcripts from the previous stage
    sri_gabon_conv_supervision=local/src/sri_gabon_conv_best_wer_transcripts.txt
    sri_gabon_read_supervision=local/src/sri_gabon_read_best_wer_transcripts.txt
    answers_supervision=local/src/yaounde_answers_best_wer_transcripts.txt

    rm -Rf $tmp_dir/sri_gabon_conv_*

    # directory for semi supervised 3 training data
    data_dir=data/train_semi_supervised_3

    if [ -d $data_dir ]; then
	rm -Rf $data_dir
    fi

    mkdir -p $data_dir

    # concatenate spk2utt file from previous stage to sri_gabon_conv spk2utt
    cat \
	data/train_semi_supervised_2/spk2utt \
	data/sri_gabon_conv/spk2utt > \
	$tmp_dir/spk2utt.unsorted

    sort \
	$tmp_dir/spk2utt.unsorted > \
	$data_dir/spk2utt

    # ditto for the utt2spk file
    cat \
	data/train_semi_supervised_2/utt2spk \
	data/sri_gabon_conv/utt2spk > \
	$tmp_dir/utt2spk.unsorted

    sort \
	$tmp_dir/utt2spk.unsorted > \
	$data_dir/utt2spk

    # ditto for the wav.scp file
    cat \
	data/train_semi_supervised_2/wav.scp \
	data/sri_gabon_conv/wav.scp > \
	$tmp_dir/wav.scp.unsorted

    sort \
	$tmp_dir/wav.scp.unsorted > \
	$data_dir/wav.scp

    #  the sri_gabon conv text  file
    # here is where we use the transcripts from the previous stage 
    if [ -e $tmp_dir/text ]; then
	rm $tmp_dir/text
    fi

    cat \
	data/train_semi_supervised_2/text \
	$sri_gabon_conv_supervision > \
	$tmp_dir/text

    sort  \
	$tmp_dir/text > \
	$data_dir/text

    # use the best sgmm transcripts as reference for testing
    cp \
	data/answers/text \
	data/answers/text.3

    cp \
	$answers_supervision \
	data/answers/text

    # ditto for the best sri_gabon read transcripts
    cp \
	data/sri_gabon_read/text \
	data/sri_gabon_read/text.2

    cp \
	$sri_gabon_read_supervision \
	data/sri_gabon_read/text

    # ditto for the best sri_gabon_conv transcripts
    cp \
	data/sri_gabon_conv/text \
	data/sri_gabon_conv/text.1

    cp \
	$sri_gabon_conv_supervision \
	data/sri_gabon_conv/text

    # extract plp pitch semi_supervised_3 features
    for part in train_semi_supervised_3 ; do
	if [ -e data/$part/cmvn.scp ]; then
	    rm data/$part/cmvn.scp
	fi

	steps/make_plp_pitch.sh \
	    --cmd "$train_cmd" \
	    --nj $train_nj \
	    data/$part \
	    exp/make_plp_pitch/$part \
	    $plp_dir || exit 1;

	utils/fix_data_dir.sh \
	    data/$part || exit 1;

	steps/compute_cmvn_stats.sh \
	    data/$part \
	    exp/make_plp/$part \
	    $plp_dir || exit 1;

	utils/fix_data_dir.sh \
	    data/$part || exit 1;
    done
fi

# train  monophones and 4 passes of triphones on semi supervised 3

if [ $stage -le 14 ]; then
    steps/train_mono.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/mono_semi_supervised_3 || exit 1;

    # align with monophones trained on semi supervised 3
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/mono_semi_supervised_3 \
	exp/mono_semi_supervised_3_ali || exit 1;

    # train tri1 on semi supervised 3 data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri1 \
	$numGaussTri1 \
	data/train_semi_supervised_3 \
	data/lang \
	exp/mono_semi_supervised_3_ali \
	exp/tri1_semi_supervised_3 || exit 1;

    # align with tri1 models trained on semi supervised 3 data
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri1_semi_supervised_3 \
	exp/tri1_semi_supervised_3_ali || exit 1;

    # train tri2 models on semi supervised 3 data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri2 \
	$numGaussTri2 \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri1_semi_supervised_3_ali \
	exp/tri2_semi_supervised_3 || exit 1;

    # align with tri2 models trained on semi supervised 3 data
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri2_semi_supervised_3 \
	exp/tri2_semi_supervised_3_ali || exit 1;

    # train tri3 models on semi supervised 3 data
    steps/train_deltas.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesTri3 \
	$numGaussTri3 \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri2_semi_supervised_3_ali \
	exp/tri3_semi_supervised_3 || exit 1;

    # align with tri3 models trained on semi supervised 3 data
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri3_semi_supervised_3 \
	exp/tri3_semi_supervised_3_ali || exit 1;

    # train tri4 with lda and mllt on semi supervised 3 data
    steps/train_lda_mllt.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesMLLT \
	$numGaussMLLT \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri3_semi_supervised_3_ali \
	exp/tri4_semi_supervised_3 || exit 1;

    # align with tri4 models trained on semi supervised 3 data
    steps/align_si.sh \
	--boost-silence $boost_sil \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri4_semi_supervised_3 \
	exp/tri4_semi_supervised_3_ali || exit 1;
fi

if [ $stage -le 15 ]; then
    # speaker adaptive training tri5 with semi supervised 3 data
    steps/train_sat.sh \
	--boost-silence $boost_sil \
	--cmd "$train_cmd" \
	$numLeavesSAT \
	$numGaussSAT \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri4_semi_supervised_3_ali \
	exp/tri5_semi_supervised_3 || exit 1;

    # start SGMM training
    echo "Starting exp/tri5_semi_supervised_3_ali on" `date`
    # do fmmlr alignment with tri5 models trained on semi supervised 3 data
    steps/align_fmllr.sh \
	--boost-silence 0.125 \
	--nj $train_nj \
	--cmd "$train_cmd" \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri5_semi_supervised_3 \
	exp/tri5_semi_supervised_3_ali || exit 1;

    # train a universal background model ubm5 on semi supervised 3 data
    steps/train_ubm.sh \
	--cmd "$train_cmd" \
	$numGaussUBM \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri5_semi_supervised_3_ali \
	exp/ubm5_semi_supervised_3 || exit 1;

    # train sgmm on semi supervised 3 data
    steps/train_sgmm2.sh \
	--cmd "$train_cmd" \
	$numLeavesSGMM \
	$numGaussSGMM \
	data/train_semi_supervised_3 \
	data/lang \
	exp/tri5_semi_supervised_3_ali \
	exp/ubm5_semi_supervised_3/final.ubm \
	exp/sgmm5_semi_supervised_3 || exit 1;

    #start discriminative SGMM training with alignment on semi supervised 3 data
    steps/align_sgmm2.sh \
	--nj $train_nj \
	--cmd "$train_cmd" \
	--transform-dir exp/tri5_semi_supervised_3_ali \
	--use-graphs true \
	--use-gselect true \
	data/train_semi_supervised_3 \
	data/lang \
	exp/sgmm5_semi_supervised_3 \
	exp/sgmm5_semi_supervised_3_ali || exit 1;

    # make denoninator lattices with semi supervised 3 data
    steps/make_denlats_sgmm2.sh \
	--nj $train_nj \
	--sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
	--beam 10.0 \
	--lattice-beam 6 \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_3_ali \
	data/train_semi_supervised_3 \
	data/lang \
	exp/sgmm5_semi_supervised_3_ali \
	exp/sgmm5_semi_supervised_3_denlats || exit 1;

    # train with boosted mmi on semi supervised 3 data
    steps/train_mmi_sgmm2.sh \
	--cmd "$train_cmd" \
	"${sgmm_mmi_extra_opts[@]}" \
	--drop-frames true \
	--transform-dir exp/tri5_semi_supervised_3_ali \
	--boost 0.1 \
	data/train_semi_supervised_3 \
	data/lang \
	exp/sgmm5_semi_supervised_3_ali \
	exp/sgmm5_semi_supervised_3_denlats \
	exp/sgmm5_semi_supervised_3_mmi_b0.1 || exit 1;
fi

if [ $stage -le 16 ]; then
    # make fst for semi supervised 3 trained tri5 models
    utils/mkgraph.sh \
	data/lang  \
	exp/tri5_semi_supervised_3 \
	exp/tri5_semi_supervised_3/graph || exit 1;

    # decode test data with tri5 models trained on semi supervised 3
    # to get transforms
    steps/decode_fmllr.sh \
	--nj $nj  \
	exp/tri5_semi_supervised_3/graph  \
	data/test \
	exp/tri5_semi_supervised_3/decode_test || exit 1;

    #ditto for answers
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised_3/graph \
	data/answers \
	exp/tri5_semi_supervised_3/decode_answers || exit 1;

    #ditto for central accord read data
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised_3/graph \
	data/sri_gabon_read \
	exp/tri5_semi_supervised_3/decode_sri_gabon_read || exit 1;

    #ditto for central accord conversational speech data
    steps/decode_fmllr.sh \
	--nj $nj \
	--cmd "$decode_cmd" \
	exp/tri5_semi_supervised_3/graph \
	data/sri_gabon_conv \
	exp/tri5_semi_supervised_3/decode_sri_gabon_conv || exit 1;

    # make sgmm decoding graph to decode with semi supervised 3 trained models
    utils/mkgraph.sh \
	data/lang \
	exp/sgmm5_semi_supervised_3 \
	exp/sgmm5_semi_supervised_3/graph || exit 1;

    # decode test with sgmm models trained on semi supervised 3 data
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_3/decode_test \
	exp/sgmm5_semi_supervised_3/graph \
	data/test \
	exp/sgmm5_semi_supervised_3/decode_test || exit 1;

    ditto for answers
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_3/decode_answers \
	exp/sgmm5_semi_supervised_3/graph \
	data/answers \
	exp/sgmm5_semi_supervised_3/decode_answers || exit 1;

    # ditto for central accord read
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_3/decode_sri_gabon_read \
	exp/sgmm5_semi_supervised_3/graph \
	data/sri_gabon_read \
	exp/sgmm5_semi_supervised_3/decode_sri_gabon_read || exit 1;

# ditto for central accord conversational data
    steps/decode_sgmm2.sh \
	--config conf/decode.config \
	--nj $nj \
	--cmd "$decode_cmd" \
	--transform-dir exp/tri5_semi_supervised_3/decode_sri_gabon_conv \
	exp/sgmm5_semi_supervised_3/graph \
	data/sri_gabon_conv \
	exp/sgmm5_semi_supervised_3/decode_sri_gabon_conv || exit 1;

    # decode test data with sgmm boosted mmi models trained on semi supervised 3
    # and lattice rescoring
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised_3/decode_test \
	    data/lang \
	    data/test \
	    exp/sgmm5_semi_supervised_3/decode_test \
	    exp/sgmm5_semi_supervised_3_mmi_b0.1/decode_test_it$iter &
    done  

    # ditto for answers
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised_3/decode_answers \
	    data/lang \
	    data/answers \
	    exp/sgmm5_semi_supervised_3/decode_answers \
	    exp/sgmm5_semi_supervised_3_mmi_b0.1/decode_answers_it$iter &
    done  

    # ditto for central accord read data
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised_3/decode_sri_gabon_read \
	    data/lang \
	    data/sri_gabon_read \
	    exp/sgmm5_semi_supervised_3/decode_sri_gabon_read \
	    exp/sgmm5_semi_supervised_3_mmi_b0.1/decode_sri_gabon_read_it$iter &
    done  

    # ditto for central accord conversational
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh \
	    --cmd "$decode_cmd" \
	    --iter $iter \
	    --transform-dir exp/tri5_semi_supervised_3/decode_sri_gabon_conv \
	    data/lang \
	    data/sri_gabon_conv \
	    exp/sgmm5_semi_supervised_3/decode_sri_gabon_conv \
	    exp/sgmm5_semi_supervised_3_mmi_b0.1/decode_sri_gabon_conv_it$iter &
    done  
