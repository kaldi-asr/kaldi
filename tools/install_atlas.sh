#!/bin/bash

# You would typically only have to run this if the ATLAS libraries are not
# already installed on your system... i.e. only after you have tried to
# do ./configure in ../src and it has failed.

# This script tries to install ATLAS-- the install script is supposed to be
# pretty system independent, but try to run it on the same machine type as the
# one you intend to use the library on-- it may produce binaries that use
# CPU dependent instructions.  If you want to use the Kaldi toolkit on a cluster
# that has different machine types, you will probably have to get your sysadmin
# involved-- either to find the "lowest common denominator" machine type, or
# to install dynamic libraries somewhere... but note that the "configure" script
# isn't currently set up to work with dynamically linked libraries.  You'd have
# to mess with ../src/kaldi.mk yourself in order to get this to work (adding
# probably -lclapack -lcblas -latalas -lf77blas, or something like that...
# let us know what worked.)

if [ ! -f atlas3.8.3.tar.gz ]; then
  wget -T 10 -t 3 http://sourceforge.net/projects/math-atlas/files/Stable/3.8.3/atlas3.8.3.tar.gz || exit 1;
fi

tar -xvzf atlas3.8.3.tar.gz  || exit 1;

cd ATLAS
mkdir build # you should probably have a name that reflects OS, CPU, etc... but this is fine
cd build

../configure --prefix=`pwd` || exit 1;
make || exit 1;
make check || exit 1;
# make time
mkdir install
make install DESTDIR=`pwd`/install || exit 1;


