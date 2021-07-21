#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0


# Download data 

data=$1
url=$2
part=$3

if [ $# -ne 3 ];then
	echo "Usage: $0 <data-base> <url-base> <corpus-part>"
	echo "e.g.: $0 data/aishell www.openslr.org/33 data_aishell.tar.gz"
fi

if [ ! -d "$data" ]; then
	mkdir -p $data
fi

if [ -z "$url" ]; then
	echo "$0: empty URL base."
	exit 1;
fi

full_url=$url/$part
echo "$full_url"
if [ ! -f $data/$part ]; then
	echo "$0: downloading data from $full_url. This may take some time, please be patient."
	cd $data
	if ! wget --no-check-certificate $full_url;then
		echo "$0: error executing wget $full_url"
		exit 1;
	fi
fi

cd $data

if ! tar -xvzf $part;then
	echo "$0: error un-tarring archive $data/$part"
	exit 1;
fi

echo "$0: Successfully downloaded and un-tarred $data/$part"
exit 1;

