#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
[ "$(basename $PWD)" != "fairseq_ltlm" ] && echo "run fairseq_ltlm/setup.sh in fairseq_ltlm dir!" && exit 1
set -e 

false && {
if [ ! -d anaconda ] ; then
	echo "Install anaconda"
	installer=Miniconda3-py38_4.8.3-Linux-x86_64.sh
	wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 20 https://repo.anaconda.com/miniconda/$installer
	bash $installer -b -p anaconda/
	rm $installer
fi
source anaconda/bin/activate

echo "Install fairseq"
(
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

#echo "Install apex"
#[ ! -d apex ] && git clone https://github.com/NVIDIA/apex
#cd apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#		  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#		    --global-option="--fast_multihead_attn" ./
)
}
echo "Install aditional packages"
(
pip install -r ltlm/requirements.txt
)

echo "Making extra kaldi tools"
(
cd kaldi_utils
make 
)


echo "Done. Don't foget to activate conda env ( source fairseq_ltlm/anaconda/bin/activate ) before running recipe"
exit 0
