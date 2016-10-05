#!/bin/bash

. ./path.sh

pm_type="entropy"
# dnn_dir=/home/bmeyer/exp/neural_nets/constantin_OLSA_multicond_mfcc_tri6b_dnn_smbr_i4
dnn_dir=/home/bmeyer/a4/exp/tri4a_multi_dnn
langdir=$1
. utils/parse_options.sh

wavefile=$2
outdir=$3 # dir to write the pm scores for wavfile
phonemapdir=$4
mkdir -p $outdir

# tmpdir=$(mktemp -d -u); mkdir -p $tmpdir
#tmpdir=`pwd`/tmp
#mkdir -p $tmpdir

# trap "echo \"# Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT

# wavefile to features
name=$(basename $wavefile .wav)
tmpdir=`pwd`/tmp_${name}
echo "tmpdir is $tmpdir"
mkdir -p $tmpdir

echo "######### Started processing $name ########## @" `date`

kaldi_data_dir=$tmpdir/data/$name; mkdir -p $kaldi_data_dir
echo $name" "$wavefile > $kaldi_data_dir/wav.scp
echo $name" "$name > $kaldi_data_dir/utt2spk
echo $name" "$name > $kaldi_data_dir/spk2utt

# steps/make_mfcc.sh --nj 1 --mfcc-config /home/bmeyer/push_forward/mfcc.conf $kaldi_data_dir $kaldi_data_dir/log $kaldi_data_dir/data || exit 1;
steps/make_fbank.sh --nj 1 --fbank-config /home/bmeyer/a4/conf/fbank.conf $kaldi_data_dir $kaldi_data_dir/log $kaldi_data_dir/data || exit 1;
steps/compute_cmvn_stats.sh $kaldi_data_dir $kaldi_data_dir/log $kaldi_data_dir/data || exit 1;

# forward pass features to get posteriors
posterior_dir=$tmpdir/posteriors/$name; mkdir -p $posterior_dir

steps/nnet/make_bn_feats.sh --nj 1 --remove-last-components 0 \
  $posterior_dir $kaldi_data_dir $dnn_dir $posterior_dir/log $posterior_dir/data || exit 1;
# convert posterior file from ark to htk  
copy-feats-to-htk --output-dir=$posterior_dir/data --output-ext=htk scp:$posterior_dir/feats.scp
copy-feats-to-htk --output-dir=$posterior_dir/data --output-ext=fbank scp:$kaldi_data_dir/feats.scp
# from posteriors to performance monitoring measures
if [ $pm_type == "entropy" ]; then
  python utils/multi-stream/pm_utils/compute_entropy.py $posterior_dir/feats.scp $outdir/${name}.pklz || exit 1
  python utils/multi-stream/pm_utils/dicts2txt.py $outdir/${name}.pklz $outdir/${name}.txt || exit 1
fi

# This has to be done only once
#echo Creating mapping from context-dependent triphones to monophones
#mkdir -p $tmpdir/phone_mappings
#utils/phone_post/create_pdf_to_phone_map.sh ${langdir} $dnn_dir/final.mdl $tmpdir/phone_mappings

echo Transforming to a monophone set
transform-nnet-posteriors --pdf-to-pseudo-phone=$phonemapdir/pdf_to_pseudo_phone_bernd.txt ark:${tmpdir}/posteriors/$name/data/raw_bnfea_${name}.1.ark  ark:${tmpdir}/posteriors/$name/data/monophone_bnfea_${name}.1.ark
echo checkout ${tmpdir}/posteriors/$name/data/monophone_bnfea_test.1.ark

echo $posterior_dir/feats_mono.scp
# cat ${posterior_dir}/feats.scp|sed s/'raw'/'monophone'/ > $posterior_dir/feats_mono.scp
# copy-feats-to-htk --output-dir=$posterior_dir/data --output-ext=monophone scp:$posterior_dir/feats_mono.scp
copy-feats-to-htk --output-dir=$posterior_dir/data --output-ext=mph ark:${tmpdir}/posteriors/$name/data/monophone_bnfea_${name}.1.ark
# echo "######### Finished processing $name ########## @" `date`
# echo "######### Did not remove temporary data #####"
exit 0;

