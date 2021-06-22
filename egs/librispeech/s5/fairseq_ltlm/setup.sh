#!/bin/bash
[ "$(basename $PWD)" != "fairseq_ltlm" ] && echo "run fairseq_ltlm/setup.sh in fairseq_ltlm dir!" && exit 1
set -e 

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
#)

echo "Done. Don't foget to activate conda env ( source anaconda/bin/activate ) before running recipe"
exit 0
