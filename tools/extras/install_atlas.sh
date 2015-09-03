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

# note: to get this install process to work you have to make
# sure CPU throttling is disabled.  To do this, on some systems
# you can edit
# /etc/sysconfig/cpuspeed  
# to set GOVERNOR=performance
# On others you can do
#  sudo cpufreq-selector performance
# You may under some circumstances have to specify -b 32 to the configure
# script (e.g. if you are compiling Kaldi in 32-bit on a 64-bit CPU).

if [ ! -f atlas3.10.0.tar.bz2 ]; then
  wget -T 10 -t 3 http://sourceforge.net/projects/math-atlas/files/Stable/3.10.0/atlas3.10.0.tar.bz2 || exit 1;
fi

tar -xvjf atlas3.10.0.tar.bz2  || exit 1;

cd ATLAS
mkdir build # you should probably have a name that reflects OS, CPU, etc... but this is fine
cd build


# sometimes the -b 32 option can be helpful to "configure"
# when it's on a 64-bit CPU but a 32-bit OS.  It won't hurt
# if it's not a 64-bit CPU.
x=`uname -a | awk '{print $(NF-1)}'`
if [ "$x" == "i686" -o "$x" == "x86" ]; then
  opt="-b 32"
fi

../configure $opt --prefix=`pwd`/install || exit 1;
make -j 2 || exit 1;
make check -j 2 || exit 1;
make install || exit 1;


