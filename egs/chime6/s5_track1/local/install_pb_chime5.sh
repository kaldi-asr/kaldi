#!/usr/bin/env bash

# Installs pb_chime5
# miniconda should be installed in $HOME/miniconda3/ 

miniconda_dir=$HOME/miniconda3/

if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run 'tools/extras/install_miniconda.sh" && exit 1;
fi

git clone https://github.com/fgnt/pb_chime5.git
cd pb_chime5
# Download submodule dependencies  # https://stackoverflow.com/a/3796947/5766934
git submodule init  
git submodule update

$miniconda_dir/bin/python -m pip install cython
$miniconda_dir/bin/python -m pip install pymongo
$miniconda_dir/bin/python -m pip install fire
$miniconda_dir/bin/python -m pip install -e pb_bss/
$miniconda_dir/bin/python -m pip install -e .
