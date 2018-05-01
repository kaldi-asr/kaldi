#!/bin/bash

dir=

if [ $# -ne 1 ]; then
	echo "install_jieba.sh <target-dir>"
	echo " e.g install_jieba.sh local/jieba"
	exit 1;
fi

dir=$1

git clone https://github.com/fxsjy/jieba.git $dir
cd $dir
sudo python setup.py install
cd -
