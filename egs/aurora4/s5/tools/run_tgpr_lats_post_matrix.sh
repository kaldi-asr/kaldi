#!/bin/bash

#input is lat.1.gz, output is phone posts on lats in matrix,
#support for only 1 lat.gz in latdir, and only 1 utt in this lats
#using 1best path

if [ $# != 4 ];then
	echo "Usage: <mdl-dir> <lat-dir> <out-dir> <phone_map-dir>"
exit 1;
fi

. ./path.sh
. ./cmd.sh

mdl=$1/final.mdl
latdir=$2
#lat="ark:gunzip -c $2/lat.1.gz|"
outdir=$3
phonedir=$4

mkdir -p $outdir/matrix

echo "$1 $2 $3 $4"

if [ -f $2/num_jobs ]; then
  nj=`cat $2/num_jobs 2>/dev/null`;
  echo "Totally contains $nj lats"
  else
echo "$num_jobs need to be specified" 
exit 1;
fi

total_phonemes=`cat $phonedir/phone_id_map_bernd_matlab.map | wc -l`
if [ ! -d $outdir ]; then
	mkdir -p $outdir
fi

step=0
for i in `seq $nj`; do
lat="ark:gunzip -c $2/lat.$i.gz|"
echo "lat is $lat"
if [ $step -le 1 ]; then
lattice-to-post --acoustic-scale=0.1 "$lat" ark:-| \
post-to-pdf-post $mdl ark:- ark,t:$outdir/lat_${i}.lats.pdf.post || exit 1;

lattice-1best --acoustic-scale=0.1 "$lat" ark:- | \
nbest-to-linear ark:- ark,t:$outdir/lat_${i}_1best.ali ark,t:$outdir/lat_${i}_1best.tra || exit 1;

ali-to-pdf $mdl ark:$outdir/lat_${i}_1best.ali ark,t:$outdir/lat_${i}_1best.ali.pdf 
get-post-on-ali ark:$outdir/lat_${i}.lats.pdf.post ark,s,cs:$outdir/lat_${i}_1best.ali.pdf ark,t:$outdir/lat_${i}_1best.pdf.post || exit 1;
fi

tools/pdf_post_to_phone_post.pl $outdir/lat_${i}_1best.ali.pdf $phonedir/pdf_to_pseudo_phone_bernd.txt $phonedir/phone_id_map_bernd_matlab.map $outdir/lat_${i}_1best.phones.id_bernd || exit 1
#int2sym.pl $phonedir/phone_id_map_bernd_matlab.map $outdir/lat_${i}_1best.phones.id_bernd > $outdir/lat_${i}_1best.phones_bernd.txt
#sym2int.pl phone_mappings_bernd_nnet1/phone_id_map_bernd_matlab.map $outdir/lat_${i}_1best.phones.txt > $outdir/lat_${i}_1best.phones_bernd.id
tools/lats_convert_to_phone_post_matrix.pl $outdir/lat_${i}_1best.phones.id_bernd $outdir/lat_${i}_1best.pdf.post $outdir/matrix $total_phonemes || exit 1

done

echo "Succeed preparing phone post matrix"




