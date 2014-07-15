#!/usr/bin/env bash

. cmd.sh

split=dev
data_dir=data/dev
decode_dir=exp/sgmm2x_6a/decode_dev/
lang_dir=data/lang

# Create the STM file
# Always create this file before creating the CTM files so that
# channel numbers are properly created. 
if [ ! -f $data_dir/stm ]; then
    /export/a11/guoguo/babel/103-bengali-limitedLP.official/local/prepare_stm.pl $data_dir
fi

# Create the CTM file
steps/get_ctm.sh $data_dir $lang_dir $decode_dir

# Make sure that channel markers match
#sed -i "s:\s.*_fsp-([AB]): \1:g" data/dev/stm
#ls exp/tri5a/decode_dev/score_*/dev.ctm | xargs -I {} sed -i -r 's:fsp\s1\s:fsp A :g' {}
#ls exp/tri5a/decode_dev/score_*/dev.ctm | xargs -I {} sed -i -r 's:fsp\s2\s:fsp B :g' {}

# Get the environment variables
. /export/babel/data/software/env.sh

# Start scoring
/export/a11/guoguo/babel/103-bengali-limitedLP.official/local/score_stm.sh $data_dir $lang_dir \
    $decode_dir

# Print a summary of the result
grep "Percent Total Error" $decode_dir/score_*/$split.ctm.dtl
