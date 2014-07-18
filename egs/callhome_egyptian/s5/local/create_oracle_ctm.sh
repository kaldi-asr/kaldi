#!/usr/bin/env bash

# No sanity checks here, they need to be added

data=data/test
dir=exp/sgmm2x_6a_mmi_b0.2/decode_test_fmllr_it1
lang=data/lang
LMWT=10

[ -f ./path.sh ] && . ./path.sh

mkdir -p $dir/oracle

cmd=run.pl
filter_cmd="utils/convert_ctm.pl $data/segments $data/reco2file_and_channel"
name=`basename $data`;
model=$dir/../final.mdl # assume model one level up from decoding dir.
symTable=$lang/words.txt

if [ ! -f $dir/oracle/oracle.lat.gz ]; then
    cat $data/text | utils/sym2int.pl -f 2- --map-oov [oov] $symTable | \
        lattice-oracle --write-lattices="ark:|gzip -c > $dir/oracle/oracle.lat.gz" \
            "ark:gunzip -c $dir/lat.*.gz|" ark:- ark:- > /dev/null 2>&1
fi
        
lattice-align-words $lang/phones/word_boundary.int $model \
    "ark:gunzip -c $dir/oracle/oracle.lat.gz|" ark:- | \
    lattice-1best --lm-scale=$LMWT ark:- ark:- | nbest-to-ctm ark:- - | \
    utils/int2sym.pl -f 5 $lang/words.txt | \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
        > $dir/oracle/$name.ctm
