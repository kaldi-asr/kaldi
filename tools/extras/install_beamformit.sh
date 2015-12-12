#!/bin/bash

#Installs beamformit from the location https://github.com/xanguera/BeamformIt

# libsndfile needed by beamformit,
  wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
  tar xvf libsndfile-1.0.25.tar.gz 
  ( 
  cd libsndfile-1.0.25;
  ./configure --prefix=$PWD/libsndfile-1.0.25/
  make
  make install
  );

  git clone https://github.com/xanguera/BeamformIt
  (
  cd BeamformIt
  cmake . 
  make
  );
