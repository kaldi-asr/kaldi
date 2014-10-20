#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

# No sanity checks here, they need to be added

data=data/callhome_test
dir=exp/tri5a/decode_callhome_test
lang=data/lang
LMWT=13

[ -f ./path.sh ] && . ./path.sh

cmd=run.pl
filter_cmd="utils/convert_ctm.pl $data/segments $data/reco2file_and_channel"
name=`basename $data`;
model=$dir/../final.mdl # assume model one level up from decoding dir.
symTable=$lang/words.txt

if [ ! -f $dir/oracle/oracle.lat.gz ]; then
    cat $data/text | utils/sym2int.pl --map-oov [oov] -f 2- $symTable | \
        lattice-oracle --write-lattices="ark:|gzip -c > $dir/oracle/oracle.lat.gz" \
            "ark:gunzip -c $dir/lat.*.gz|" ark:- ark:- > /dev/null 2>&1
fi
        
lattice-align-words $lang/phones/word_boundary.int $model \
    "ark:gunzip -c $dir/oracle/oracle.lat.gz|" ark:- | \
    lattice-1best --lm-scale=$LMWT ark:- ark:- | nbest-to-ctm ark:- - | \
    utils/int2sym.pl -f 5 $lang/words.txt | \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
        > $dir/oracle/$name.ctm
