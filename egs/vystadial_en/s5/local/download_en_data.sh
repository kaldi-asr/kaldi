#!/bin/bash
# Copyright Ondrej Platek Apache 2.0

DATA_ROOT=$1

url=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0023-4671-4/data_voip_en.tgz
# This might be faster:
#url=http://www.openslr.org/resources/6/data_voip_en.tgz
name=data_voip_en
extract_file=205859

mkdir -p $DATA_ROOT

if [ ! -f $DATA_ROOT/${name}.tgz ] ; then
    wget $url -O $DATA_ROOT/${name}.tgz || exit 1
    echo "Data successfully downloaded"
fi

if [[ ! -d $DATA_ROOT/$name && -e $DATA_ROOT/$name ]] ; then
    echo "The $DATA_ROOT/$name is not a directory and we cannot extract the data!"
    exit 1;
fi

if [ ! -d $DATA_ROOT/$name ] ; then
    mkdir $DATA_ROOT/$name
    tar xfv $DATA_ROOT/${name}.tgz -C $DATA_ROOT | \
    while read line; do
        x=$((x+1))
        echo -en "$x extracted from $extract_file files.\r"
    done
fi

if [ -d $DATA_ROOT/$name ] ; then
    echo "Checking if data extracted correctly"
    num_files=`find $DATA_ROOT/$name -name '*' | wc -l`
    if [ ! $num_files -eq $extract_file ] ; then
        echo "Data extraction failed! Extracted $num_files instead of $extract_file"
        exit 1;
    fi
    echo "It seams that data are extracted correctly"
fi

pushd $DATA_ROOT
    for t in test train dev ; do
        ln -s $name/$t
    done
    ln -s $name/arpa_bigram arpa-bigram
popd
