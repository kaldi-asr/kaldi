#!/bin/bash

pushd db

if [ ! -f TEDLIUM_release1.tar.gz ]; then
    wget http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz
    tar xf TEDLIUM_release1.tar.gz
fi

if [ ! -f cmusphinx-5.0-en-us.lm.gz ]; then
    wget \
        http://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English%20Generic%20Language%20Model/cmusphinx-5.0-en-us.lm.gz/download \
        -O cmusphinx-5.0-en-us.lm.gz
fi

popd
