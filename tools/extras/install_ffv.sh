#!/bin/bash

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

mkdir -p pitch_trackers
cd pitch_trackers

echo "Installing a package for FFV feature extraction."

if [ -s ffv-1.0.1.tar.gz ]; then
  echo "*ffv-1.0.1.tar.gz already exists, not getting it."
else
  ! wget -t 2 http://www.cs.cmu.edu/~kornel/software/ffv-1.0.1.tar.gz && \
    echo "Error wgetting ffv-1.0.1.tar.gz" && exit 1;
fi

if [ -d ffv-1.0.1 ]; then
  echo "*It looks like ffv-1.0.1.tar.gz has already been unpacked, not unpacking it."
else 
  ! tar -zxvf ffv-1.0.1.tar.gz && \
  echo "Error unpacking  ffv-1.0.1.tar.gz [e.g. unpack not installed?]" && exit 1;
fi
cd ffv-1.0.1

if [ -f Makefile ]; then
  echo "Makefile already exists, no creating it."
else
  echo "Makefile does not exist, creating it."
  cat<<'EOF' > ./Makefile
  CC     = gcc
  # CFLAGS = -c -O3 -Wall -pedantic -std=c99 
  CFLAGS = -c -g -Wall -pedantic -std=c99
  LIBS   = -lm

  LIBOBJECTS = \
  	\
	windowpair.o \
	filterbank.o \
	dcorrxform.o \
	ffv.o \
	mutils.o \
	sutils.o

  all : ffv 

  ffv : ffv_main.o ${LIBOBJECTS}
	${CC} -o $@ $^ ${LIBS}

  %.o : %.c
	${CC} ${CFLAGS} $<

  clean :
    rm -f *.o ffv
EOF
chmod +x Makefile 
fi
make; 
cd ..

echo "Installing ffv package is done."
exit 0;



