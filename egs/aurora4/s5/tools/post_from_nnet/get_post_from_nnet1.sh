#!/bin/bash

. ./path.sh
. utils/parse_options.sh

pm_type="entropy"

dnndir=$1
#langdir=$2
srcdata=$2
phonemapdir=$3
postdir=$4
#outdir=$6 #bn_feats
name=$5 #name of the test set

mkdir -p $postdir
nj=4

./steps/nnet/make_bn_feats.sh --nj $nj --remove-last-components 0 $postdir $srcdata $dnndir $postdir/log $postdir/data || exit 1;

copy-feats-to-htk --output-dir=$postdir/data --output-ext=htk scp:$postdir/feats.scp
copy-feats-to-htk --output-dir=$postdir/data --output-ext=fbank scp:$srcdata/feats.scp

if [ $pm_type =="entropy" ]; then
	python utils/multi-stream/pm_utils/compute_entropy.py $postdir/feats.scp $postdir/${name}.pklz || exit 1
	python utils/multi-stream/pm_utils/dicts2txt.py $outdir/${name}.pklz $postdir/${name}.txt || exit 1
fi

for i in `seq 1 $nj`; do
transform-nnet-posteriors --pdf-to-pseudo-phone=$phonemapdir/pdf_to_pseudo_phone_bernd.txt ark:$postdir/data/raw_bnfea_${name}.$i.ark ark,t:$postdir/data/monophone_bnfea_${name}.$i.txt

copy-feats-to-htk --output-dir=$postdir/data --output-ext=mph ark,t:$postdir/data/monophone_bnfea_${name}.$i.txt
done

cat $postdir/data/monophone_bnfea_${name}.*.txt | sed 's/^\s\+0//' > $postdir/monophone_bnfea_${name}.all.matrix

exit 0;
