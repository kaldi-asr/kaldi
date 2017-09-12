#!/bin/bash 

. ./cmd.sh
. ./path.sh
stage=0

. ./utils/parse_options.sh

set -e
set -o pipefail
set u

# the location of the LDC corpus
datadir=/mnt/corpora/LDC2006S37/data

# acoustic models are trained on the heroico corpus
# testing is done on the usma corpus
# heroico consists of 2 parts: answers and recordings (recited)

answers_transcripts=$datadir/transcripts/heroico-answers.txt
recordings_transcripts=$datadir/transcripts/heroico-recordings.txt

# usma is all recited
usma_transcripts=$datadir/transcripts/usma-prompts.txt

tmpdir=data/local/tmp

# make acoustic model training  lists
if [ $stage -le 0 ]; then
    mkdir \
	-p \
	$tmpdir/heroico \
	$tmpdir/usma

    local/get_wav_list.sh \
	$datadir

    # copy waveform data and make separate lists for heroico answers and recordings 
    local/heroico_answers_copy_wav_files.pl \
	$answers_transcripts

    utils/fix_data_dir.sh \
	$tmpdir/heroico/answers

    local/heroico_recordings_copy_wav_files.pl \
    	$recordings_transcripts

    utils/fix_data_dir.sh \
	$tmpdir/heroico/recordings

    # consolidate heroico lists
    mkdir -p $tmpdir/heroico/lists

    for x in wav.scp utt2spk text; do
	cat \
	    $tmpdir/heroico/answers/$x \
	    $tmpdir/heroico/recordings/$x \
	    | \
	    sort \
		-k1,1 \
		-u \
		> \
		$tmpdir/heroico/lists/$x
    done

    utils/fix_data_dir.sh \
	$tmpdir/heroico/lists

    # copy waveform data and make separate lists for usma native and nonnative
    local/usma_native_copy_wav_files.pl \
	$usma_transcripts

    local/usma_nonnative_copy_wav_files.pl \
	$usma_transcripts
    for n in native nonnative; do
	mkdir -p $tmpdir/usma/$n/lists

	for x in wav.scp utt2spk text; do
	    sort \
		$tmpdir/usma/$n/$x \
		> \
		$tmpdir/usma/$n/lists/$x
	done

	utils/fix_data_dir.sh \
	    $tmpdir/usma/$n/lists
    done

    mkdir -p data/train
    mkdir -p $tmpdir/lists

    # get training lists
    for x in wav.scp utt2spk text; do
	cat \
	    $tmpdir/heroico/answers/${x} \
	    $tmpdir/heroico/recordings/${x} \
	    > \
	    $tmpdir/lists/$x

	sort \
	    $tmpdir/lists/$x \
	    > \
	    data/train/$x
    done

    utils/utt2spk_to_spk2utt.pl \
	data/train/utt2spk \
	| \
	sort \
	    > \
	    data/train/spk2utt

    utils/fix_data_dir.sh \
	data/train


# make testing  lists

    mkdir \
	-p \
	data/test \
	data/native \
	data/nonnative \
	$tmpdir/usma/lists

    for x in wav.scp text utt2spk; do
	# get testing lists
	for n in native nonnative; do
	    cat \
		$tmpdir/usma/$n/lists/$x \
		>> \
		$tmpdir/usma/lists/$x
	done

	sort \
	    $tmpdir/usma/lists/$x \
	    > \
	    data/test/$x

	for n in native nonnative; do
	    sort \
		$tmpdir/usma/$n/$x \
		> \
		data/$n/$x
	done
    done

    # spk2utt
    for n in native nonnative test; do
	utils/utt2spk_to_spk2utt.pl \
	    data/$n/utt2spk \
	    | \
	    sort \
		> \
		data/$n/spk2utt

	utils/fix_data_dir.sh \
	    data/$n
    done
fi

if [ $stage -le 1 ]; then
    # prepare a dictionary
    mkdir -p data/local/dict

    local/prepare_dict.sh
fi

if [ $stage -le 2 ]; then
    # prepare the lang directory
    utils/prepare_lang.sh \
	data/local/dict \
	"<UNK>" \
	data/local/lang \
	data/lang   || exit 1;
fi

if [ $stage -le 3 ]; then
    # extract acoustic features
    mkdir -p exp

    for fld in native nonnative test train; do
	if [ -e data/$fld/cmvn.scp ]; then
	    rm data/$fld/cmvn.scp
	fi

	steps/make_mfcc.sh \
	    --cmd run.pl \
	    --nj 4 \
	    data/$fld \
	    exp/make_mfcc/$fld \
	    mfcc || exit 1;

	utils/fix_data_dir.sh \
	    data/$fld || exit 1;

	steps/compute_cmvn_stats.sh \
	    data/$fld \
	    exp/make_mfcc\
	    mfcc || exit 1;

	utils/fix_data_dir.sh \
	    data/$fld || exit 1;
	done
fi

if [ $stage -le 4 ]; then
    # prepare lm on training transcripts
    local/prepare_lm.sh

    # make lm fst
    mkdir -p data/local/lm

    gunzip language_models/lm_threegram.arpa.gz 

    # find out of vocabulary words
    utils/find_arpa_oovs.pl \
	data/lang/words.txt \
	language_models/lm_threegram.arpa \
	> \
	data/lang/oovs_3g.txt || exit 1;

    # make an fst out of the lm
    arpa2fst \
	language_models/lm_threegram.arpa \
	> \
	data/lang/lm_3g.fst || exit 1;

    # remove out of vocabulary arcs
    fstprint 	\
	data/lang/lm_3g.fst \
	| \
	utils/remove_oovs.pl 	    \
	    data/lang/oovs_3g.txt \
	    > \
	    data/lang/lm_3g_no_oovs.txt

    # replace epsilon symbol with \#0
    utils/eps2disambig.pl \
	< \
	data/lang/lm_3g_no_oovs.txt \
	| \
	utils/s2eps.pl \
	    > \
	    data/lang/lm_3g_with_disambig_symbols_without_s.txt

    # binarize the fst
    fstcompile \
	--isymbols=data/lang/words.txt \
	--osymbols=data/lang/words.txt \
	--keep_isymbols=false \
	--keep_osymbols=false \
	data/lang/lm_3g_with_disambig_symbols_without_s.txt \
	data/lang/lm_3g_with_disambig_symbols_without_s.fst

    # make the G.fst
    fstarcsort \
	data/lang/lm_3g_with_disambig_symbols_without_s.fst \
	data/lang/G.fst

    gzip \
	language_models/lm_threegram.arpa
fi

if [ $stage -le 5 ]; then
        echo "Starting  monophone training in exp/mono on" `date`
    steps/train_mono.sh \
	--nj 4 \
	--cmd run.pl \
	data/train \
	data/lang \
	exp/mono || exit 1;

    # align with monophones
    steps/align_si.sh \
	--nj 8 \
	--cmd run.pl \
	data/train \
	data/lang \
	exp/mono \
	exp/mono_ali || exit 1;
fi

if [ $stage -le 6 ]; then
    echo "Starting  triphone training in exp/tri1 on" `date`
    steps/train_deltas.sh \
	--cmd run.pl \
	--cluster-thresh 100 \
	1500 \
	25000 \
	data/train \
	data/lang \
	exp/mono_ali \
	exp/tri1 || exit 1;

    # align with triphones
    steps/align_si.sh \
	--nj 8 \
	--cmd run.pl \
	data/train \
	data/lang \
	exp/tri1 \
	exp/tri1_ali
fi

if [ $stage -le 7 ]; then
    echo "Starting (lda_mllt) triphone training in exp/tri2b on" `date`
    steps/train_lda_mllt.sh \
	--splice-opts "--left-context=3 --right-context=3" \
	2000 \
	30000 \
	data/train \
	data/lang \
	exp/tri1_ali \
	exp/tri2b

    # align with lda and mllt adapted triphones
    steps/align_si.sh \
	--use-graphs true \
	--nj 8 \
	--cmd run.pl \
	data/train \
	data/lang \
	exp/tri2b \
	exp/tri2b_ali

    echo "Starting (SAT) triphone training in exp/tri3b on" `date`
    steps/train_sat.sh \
	--cmd run.pl \
	3100 \
	50000 \
	data/train \
	data/lang \
	exp/tri2b_ali \
	exp/tri3b

    # align with tri3b models
    echo "Starting exp/tri3b_ali on" `date`
    steps/align_fmllr.sh \
	--nj 8 \
	--cmd run.pl \
	data/train \
	data/lang \
	exp/tri3b \
	exp/tri3b_ali
fi

if [ $stage -le 8 ]; then
    # i-vector  extractor training
    local/nnet3/run_ivector_common.sh

    # train and test chain models
    local/chain/run_tdnn.sh
fi

if [ $stage -le 9 ]; then
    # evaluation
    # make decoding graph for monophones 
    utils/mkgraph.sh \
	data/lang \
	exp/mono \
	exp/mono/graph || exit 1;


    # test monophones
    for x in native nonnative test; do
	steps/decode.sh \
	    --nj 8  \
	    exp/mono/graph  \
	    data/$x \
	    exp/mono/decode_${x} || exit 1;
    done

    # test cd gmm hmm models
    # make decoding graph for tri1
    utils/mkgraph.sh \
	data/lang  \
	exp/tri1 \
	exp/tri1/graph || exit 1;

    # decode test data with tri1 models
    for x in native nonnative test; do
	steps/decode.sh \
	    --nj 8  \
	    exp/tri1/graph  \
	    data/$x \
	    exp/tri1/decode_${x} || exit 1;
    done

    #  make decoding fst for tri2b models
    utils/mkgraph.sh \
	data/lang  \
	exp/tri2b \
	exp/tri2b/graph || exit 1;

    # decode  test with tri2b models
    for x in native nonnative test; do
	steps/decode.sh \
	--nj 8  \
	exp/tri2b/graph \
	data/$x \
	exp/tri2b/decode_${x} || exit 1;
    done

    # make decoding graph for SAT models
    utils/mkgraph.sh \
	data/lang  \
	exp/tri3b \
	exp/tri3b/graph ||  exit 1;

    # decode test set with tri3b models
    for x in native nonnative test; do
	steps/decode_fmllr.sh \
	--nj 8 \
	--cmd run.pl \
        exp/tri3b/graph \
	data/$x \
        exp/tri3b/decode_${x}
    done
fi
