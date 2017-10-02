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

# location of a reference language model
lm=http://www.csl.uni-bremen.de/GlobalPhone/lm/SP.3gram.lm.gz

tmpdir=data/local/tmp

# make acoustic model training  lists
if [ $stage -le 0 ]; then
    mkdir \
	-p \
	$tmpdir/heroico \
	$tmpdir/usma

    local/get_wav_list.sh \
	$datadir

    # make separate lists for heroico answers and recordings
    # the transcripts are converted to UTF8
    export LC_ALL=en_US.UTF-8
    cat \
	$answers_transcripts \
	| \
	iconv \
	    -f ISO-8859-1 \
	    -t UTF-8 \
	| \
	sed -e s/// \
	| \
	local/heroico_answers_make_lists.pl

    utils/fix_data_dir.sh \
	$tmpdir/heroico/answers

    cat \
	$recordings_transcripts \
	| \
	iconv \
	    -f ISO-8859-1 \
	    -t UTF-8 \
	| \
	sed -e s/// \
	    | \
	local/heroico_recordings_make_lists.pl

    utils/fix_data_dir.sh \
	$tmpdir/heroico/recordings

    # consolidate heroico lists
    mkdir -p $tmpdir/heroico/lists

    for x in wav.scp utt2spk text; do
	cat \
	    $tmpdir/heroico/answers/$x \
	    $tmpdir/heroico/recordings/$x \
	    | \
	    sed -e s/// \
		| \
	    sort \
		-k1,1 \
		-u \
		> \
		$tmpdir/heroico/lists/$x
    done

    utils/fix_data_dir.sh \
	$tmpdir/heroico/lists
fi

if [ $stage -le 1 ]; then
    #  make separate lists for usma native and nonnative
    cat \
	$usma_transcripts \
	| \
	iconv -f ISO-8859-1 -t UTF-8 \
	| \
	sed -e s/// \
	    | \
	    local/usma_native_make_lists.pl

    cat \
	$usma_transcripts \
	| \
	iconv -f ISO-8859-1 -t UTF-8 \
	| \
	sed -e s/// \
	    | \
	local/usma_nonnative_make_lists.pl

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
	    | \
	    sed -e s/// \
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

if [ $stage -le 2 ]; then
    # prepare a dictionary
    mkdir -p data/local/dict
    mkdir -p data/local/tmp/dict

    # download the dictionary from openslr
    if [ ! -f data/local/tmp/dict/santiago.tar.gz ]; then
	wget \
	    -O data/local/tmp/dict/santiago.tar.gz \
	    http://www.openslr.org/resources/34/santiago.tar.gz
    fi

    if [ -e data/local/tmp/dict/santiago.tar ]; then
	rm data/local/tmp/dict/santiago.tar
    fi

    gunzip data/local/tmp/dict/santiago.tar.gz

    cd data/local/tmp/dict

    tar -xvf santiago.tar

    cd ../../../..

    local/prepare_dict.sh

    # prepare the lang directory
    utils/prepare_lang.sh \
	data/local/dict \
	"<UNK>" \
	data/local/lang \
	data/lang   || exit 1;
fi

if [ $stage -le 3 ]; then
    # prepare lm on training transcripts
    local/prepare_lm.sh

    utils/format_lm.sh \
	data/lang \
	data/local/lm/lm_threegram.arpa.gz \
	data/local/dict/lexicon.txt \
	data/lang_test_simple

    mkdir -p $tmpdir/lm
    # retrieve a reference language model
    wget \
	-O $tmpdir/lm/ES.3gram.lm.gz \
	$lm

    utils/format_lm.sh \
	data/lang \
	data/local/tmp/lm/ES.3gram.lm.gz \
	data/local/dict/lexicon.txt \
	data/lang_test_gplm
fi

if [ $stage -le 4 ]; then
    # extract acoustic features
    mkdir -p exp

    for fld in native nonnative test train; do
	if [ -e data/$fld/cmvn.scp ]; then
	    rm data/$fld/cmvn.scp
	fi

	steps/make_mfcc.sh \
	    --cmd "$train_cmd" \
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

if [ $stage -le 5 ]; then
    echo "monophone training"
    steps/train_mono.sh \
	--nj 4 \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/mono || exit 1;

    # evaluation
    (
	# make decoding graph for monophones with 2 lm
	for l in simple gplm; do
	    utils/mkgraph.sh \
		data/lang_test_${l} \
		exp/mono \
		exp/mono/graph_${l} || exit 1;

	    # test monophones
	    for x in native nonnative test; do
		steps/decode.sh \
		    --nj 8  \
		    exp/mono/graph_${l} \
		    data/$x \
		    exp/mono/decode_${x}_${l} || exit 1;
	    done
	done
    ) &
fi

if [ $stage -le 6 ]; then
    # align with monophones
    steps/align_si.sh \
	--nj 8 \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/mono \
	exp/mono_ali || exit 1;

    echo "Starting  triphone training in exp/tri1"
    steps/train_deltas.sh \
	--cmd "$train_cmd" \
	--cluster-thresh 100 \
	1500 \
	25000 \
	data/train \
	data/lang \
	exp/mono_ali \
	exp/tri1 || exit 1;

    # test cd gmm hmm models
    # make decoding graphs for tri1
    (
	for l in simple gplm; do
	    utils/mkgraph.sh \
		data/lang_test_${l} \
		exp/tri1 \
		exp/tri1/graph_${l} || exit 1;

	    # decode test data with tri1 models
	    for x in native nonnative test; do
		steps/decode.sh \
		    --nj 8  \
		    exp/tri1/graph_${l} \
		    data/$x \
		    exp/tri1/decode_${x}_${l} || exit 1;
	    done
	done
    ) &

    # align with triphones
    steps/align_si.sh \
	--nj 8 \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri1 \
	exp/tri1_ali
fi

if [ $stage -le 7 ]; then
    echo "Starting (lda_mllt) triphone training in exp/tri2b"
    steps/train_lda_mllt.sh \
	--splice-opts "--left-context=3 --right-context=3" \
	2000 \
	30000 \
	data/train \
	data/lang \
	exp/tri1_ali \
	exp/tri2b

    (
	#  make decoding FSTs for tri2b models
	for l in simple gplm; do
	    utils/mkgraph.sh \
		data/lang_test_${l} \
		exp/tri2b \
		exp/tri2b/graph_${l} || exit 1;

	    # decode  test with tri2b models
	    for x in native nonnative test; do
		steps/decode.sh \
		    --nj 8  \
		    exp/tri2b/graph_${l} \
		    data/$x \
		    exp/tri2b/decode_${x}_${l} || exit 1;
	    done
	done
    ) &

    # align with lda and mllt adapted triphones
    steps/align_si.sh \
	--use-graphs true \
	--nj 8 \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri2b \
	exp/tri2b_ali

    echo "Starting (SAT) triphone training in exp/tri3b"
    steps/train_sat.sh \
	--cmd "$train_cmd" \
	3100 \
	50000 \
	data/train \
	data/lang \
	exp/tri2b_ali \
	exp/tri3b

    # align with tri3b models
    echo "Starting exp/tri3b_ali"
    steps/align_fmllr.sh \
	--nj 8 \
	--cmd "$train_cmd" \
	data/train \
	data/lang \
	exp/tri3b \
	exp/tri3b_ali
fi

if [ $stage -le 8 ]; then
    (
	# make decoding graphs for SAT models
	for l in simple gplm; do
	    utils/mkgraph.sh \
		data/lang_test_${l} \
		exp/tri3b \
		exp/tri3b/graph_${l} ||  exit 1;

	    # decode test sets with tri3b models
	    for x in native nonnative test; do
		steps/decode_fmllr.sh \
		    --nj 8 \
		    --cmd "$decode_cmd" \
		    exp/tri3b/graph_${l} \
		    data/$x \
		    exp/tri3b/decode_${x}_${l}
	    done
	done
    ) &
fi

if [ $stage -le 9 ]; then
    # train and test chain models
    local/chain/run_tdnn.sh
fi
