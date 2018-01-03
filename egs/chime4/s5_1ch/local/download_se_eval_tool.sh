#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

wget http://bass-db.gforge.inria.fr/bss_eval/bss_eval_sources.m -O local/bss_eval_sources.m
wget https://github.com/JacobD10/SoundZone_Tools/raw/master/stoi.m -O local/stoi.m
wget https://github.com/JacobD10/SoundZone_Tools/raw/master/estoi.m -O local/estoi.m
wget 'https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200102-I!!SOFT-ZST-E&type=items' -O PESQ.zip
unzip PESQ.zip -d local/PESQ_sources
cd local/PESQ_sources/P862/Software/source
gcc  *.c -lm -o PESQ
cd ../../../../../
mv local/PESQ_sources/P862/Software/source/PESQ local/
