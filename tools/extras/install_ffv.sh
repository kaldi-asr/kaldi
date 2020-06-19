#!/usr/bin/env bash

VERSION=1.0.1

WGET=${WGET:-wget}

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

mkdir -p pitch_trackers
cd pitch_trackers

echo "Installing a package for FFV feature extraction."

if [ -s ffv-$VERSION.tar.gz ]; then
  echo "*ffv-$VERSION.tar.gz already exists, not getting it."
elif [ -d "$DOWNLOAD_DIR" ]; then
  cp -p "$DOWNLOAD_DIR/ffv-$VERSION.tar.gz" . || exit 1
else
  ! $WGET -t 2 https://www.cs.cmu.edu/~kornel/software/ffv-$VERSION.tar.gz && \
    echo "Error wgetting ffv-$VERSION.tar.gz" && exit 1;
fi

if [ -d ffv-$VERSION ]; then
  echo "*It looks like ffv-$VERSION.tar.gz has already been unpacked, not unpacking it."
else 
  ! tar -zxvf ffv-$VERSION.tar.gz && \
  echo "Error unpacking  ffv-$VERSION.tar.gz [e.g. unpack not installed?]" && exit 1;
fi
cd ffv-$VERSION

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



