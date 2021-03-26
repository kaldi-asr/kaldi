#!/usr/bin/env bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# This script downloads the official REVERB challenge SE scripts and SRMR toolbox
# This script also downloads and compiles PESQ
# please make sure that you or your institution have the license to report PESQ
# Apache 2.0

wget 'https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200102-I!!SOFT-ZST-E&type=items' -O PESQ.zip
unzip PESQ.zip -d local/PESQ_sources
rm PESQ.zip
cd local/PESQ_sources/P862/Software/source
gcc  *.c -lm -o PESQ
cd ../../../../../
mv local/PESQ_sources/P862/Software/source/PESQ local/

wget 'https://reverb2014.dereverberation.com/tools/REVERB-SPEENHA.Release04Oct.zip' -O REVERB_scores.zip
unzip REVERB_scores.zip -d local/REVERB_scores_source
rm REVERB_scores.zip

pushd local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools
perl -i -pe 's/wavread/audioread/g' prog/score_sim.m
git clone https://github.com/MuSAELab/SRMRToolbox.git
perl -i -pe 's/wavread/audioread/g' SRMRToolbox/libs/preprocess.m
perl -i -pe 's/SRMR_main/SRMR/g' prog/score_real.m
perl -i -pe 's/SRMR_main/SRMR/g' prog/score_sim.m
perl -i -pe 's/\+wb //g' prog/calcpesq.m
perl -i -pe 's/pesq_/_pesq_/g' prog/calcpesq.m
perl -n -i -e 'print unless /remove target file name/' prog/calcpesq.m
patch score_RealData.m -i ../../../score_RealData.patch -o score_RealData_new.m
mv score_RealData_new.m score_RealData.m
patch score_SimData.m -i ../../../score_SimData.patch -o score_SimData_new.m
mv score_SimData_new.m score_SimData.m
popd
