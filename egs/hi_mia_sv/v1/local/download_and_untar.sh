#!/bin/bash

data=$1
url=$2
part=$3

if [ $# -ne 3 ];then
	echo "Usage: $0 <data-base> <url-base> <corpus-part>"
	echo "e.g.: $0 data/aishell www.openslr.org/33 data_aishell"
fi

if [ ! -d "$data" ]; then
	mkdir -p $data
fi

if [ -z "$url" ]; then
	echo "$0: empty URL base."
	exit 1;
fi

full_url=$url
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

if ! tar -xvzf $part.tar.gz;then
	echo "$0: error un-tarring archive $data/$part"
	exit 1;
fi

cd -

echo "$0: Successfully downloaded and un-tarred $data/$part"
exit 0;

