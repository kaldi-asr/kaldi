#!/bin/bash

# Installs beamformit from the location https://github.com/xanguera/BeamformIt

# libsndfile needed by beamformit,
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
tar xvf libsndfile-1.0.25.tar.gz 
( 
cd libsndfile-1.0.25;
./configure --prefix=$PWD
make
make install
);

# building beamformit,
git clone https://github.com/xanguera/BeamformIt
(
cd BeamformIt
cmake -DLIBSND_INSTALL_DIR=$PWD/../libsndfile-1.0.25 .
make
);
