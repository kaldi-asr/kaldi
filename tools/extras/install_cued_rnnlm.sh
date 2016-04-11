#!/bin/bash

set  -e

if [ ! -d cuedrnnlm ]; then
  wget http://mi.eng.cam.ac.uk/projects/cued-rnnlm/cuedrnnlm.tar.gz
  tar -zxvf cuedrnnlm.tar.gz
fi

cd cuedrnnlm


if [ ! -f /usr/local/cuda/bin/nvcc ]; then
  echo This needs to be done on a machine with GPUs!
  exit 1
fi

echo nvcc found. Will start building cued-rnnlm

export PATH=$PATH:/usr/local/cuda/bin/
./build.sh

rm rnnlm
ln -s rnnlm.cued rnnlm

wget http://mi.eng.cam.ac.uk/projects/cued-rnnlm/src.evaloncpu.tar.gz
tar -zxvf src.evaloncpu.tar.gz
cd src.evaloncpu
./build.sh
cd ../
ln -s src.evaloncpu/rnnlm.eval

echo cued-rnnlm succesfully installed
