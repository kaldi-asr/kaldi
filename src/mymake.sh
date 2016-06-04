#!/bin/sh
cd ../chain || exit 1
make  || exit 1
cd ../nnet3  || exit 1
make -j 2 -B nnet-chain-example.o nnet-chain-training.o || exit 1
make || exit 1
cd ../chainbin || exit 1
make -j 2
