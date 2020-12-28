#!/usr/bin/env bash

GIT=${GIT:-git}

# Installs nara-wpe with dependencies
# miniconda should be installed in $HOME/miniconda3/

miniconda_dir=$HOME/miniconda3/

if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run 'tools/extras/install_miniconda.sh" && exit 1;
fi

$HOME/miniconda3/bin/python -m pip install soundfile
$GIT clone https://github.com/fgnt/nara_wpe.git
cd nara_wpe
$HOME/miniconda3/bin/python -m pip install --editable .
