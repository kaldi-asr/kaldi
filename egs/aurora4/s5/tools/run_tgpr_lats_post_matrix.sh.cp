#!/bin/bash

#input is lat.1.gz, output is phone posts on lats in matrix,
#support for only 1 lat.gz in latdir, and only 1 utt in this lats
#using 1best path

if [ $# != 4 ];then
	echo "Usage: <mdl-dir> <lat-dir> <out-dir> <phone_map-dir>"
exit 1;
fi

. ./path.sh

mdl=$1/final.mdl
lat="ark:gunzip -c $2/lat.1.gz|"
outdir=$3
phonedir=$4

echo "$1 $2 $3 $4"

lattice-to-post --acoustic-scale=0.1 "$lat" ark:-| \
post-to-pdf-post $mdl ark:- ark,t:$outdir/1.lats.pdf.post || exit 1;

lattice-1best --acoustic-scale=0.1 "$lat" ark,t:- | \
nbest-to-linear ark:- ark,t:$outdir/1best.ali ark,t:$outdir/1best.tra || exit 1;

ali-to-pdf $mdl ark:$outdir/1best.ali ark,t:$outdir/1best.ali.pdf 
get-post-on-ali ark:$outdir/1.lats.pdf.post ark,s,cs:$outdir/1best.ali.pdf ark,t:$outdir/1best.pdf.post || exit 1;

tools/pdf_post_to_phone_post.pl $outdir/1best.ali.pdf $phonedir/pdf_to_pseduo_phone.txt $phonedir/pseudo_phones.txt $outdir/1best.phones.id
int2sym.pl $phonedir/pseudo_phones.txt $outdir/1best.phones.id > $outdir/1best.phones.txt
sym2int.pl phone_mappings_bernd_nnet1/phone_id_map_bernd_matlab.map $outdir/1best.phones.txt > $outdir/1best.phones_bernd.id

tools/lats_convert_to_phone_post_matrix.pl $outdir/1best.phones_bernd.id $outdir/1best.pdf.post $outdir/1best.phone.post.matrix

echo "Succeed preparing phone post matrix"




