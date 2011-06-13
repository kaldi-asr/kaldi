#!/bin/bash
# This script attempts to automatically execute the instructions in INSTALL.

# (1) Install instructions for sph2pipe_v2.5.tar.gz

if ! which wget >&/dev/null; then
   echo "This script requires you to first install wget";
   exit 1;
fi

if ! which automake >&/dev/null; then
   echo "Warning: automake not installed (IRSTLM installation will not work)"
   sleep 1
fi

if ! which libtoolize >&/dev/null; then
   echo "Warning: libtoolize not installed (IRSTLM installation probably will not work)"
   sleep 1
fi


echo "****(1) Installing sph2pipe"

(
  rm sph2pipe_v2.5.tar.gz 2>/dev/null
  wget -T 10 -t 3 ftp://ftp.ldc.upenn.edu/pub/ldc/misc_sw/sph2pipe_v2.5.tar.gz

  if [ ! -e sph2pipe_v2.5.tar.gz ]; then
    echo "****download of sph2pipe_v2.5.tar.gz failed."
    exit 1
else
    tar -xovzf sph2pipe_v2.5.tar.gz || exit 1
    cd sph2pipe_v2.5
    gcc -o sph2pipe *.c -lm  || exit 1
    cd ..
fi
)
ok_sph2pipe=$?
if [ $ok_sph2pipe -ne 0 ]; then
  echo "****sph2pipe install failed."
fi


# (2) download Atlas
(
  echo "****(2) downloading ATLAS"

  wget -T 10 -t 3 http://sourceforge.net/projects/math-atlas/files/Stable/3.8.3/atlas3.8.3.tar.gz

  if [ ! -e atlas3.8.3.tar.gz ]; then
      echo "****download atlas3.8.3.tar.gz failed."
      exit 1
  else
    tar -xovzf atlas3.8.3.tar.gz ATLAS/include || exit 1
  fi
)
ok_atlas=$?
if [ $ok_atlas -ne 0 ]; then
  echo "****Download of ATLAS headers failed."
fi


# (3) download CLAPACK headers
(
  echo "****(3) downloading CLAPACK headers"

  mkdir CLAPACK_include
  cd CLAPACK_include

  for x in  clapack.h    f2c.h; do 
    wget -T 10 -t 3 http://www.netlib.org/clapack/$x; 
  done

  if [ ! -e clapack.h ] || [ ! -e f2c.h ]; then
    echo "****download clapack.h or f2c.h failed."
    cd ..
    exit 1
  fi
  cd ..
)
ok_clapack=$?
if [ $ok_clapack -ne 0 ]; then
  echo "****Download of CLAPACK headers failed."
fi


(
  # (4) Install instructions to install IRSTLM.
  # This is not needed for the basic system builds (RM has its own
  # non-ARPA LM, and WSJ comes with LMs).  So installing this may be
  # left till later, if you are in a hurry.
  echo "****(4) install IRSTLM (optional; only needed if you want to build LMs and don't already have a setup)"

  svn co https://irstlm.svn.sourceforge.net/svnroot/irstlm/trunk irstlm || exit 1

  if [ ! -e irstlm ]; then
    echo "***download of irstlm failed."
    exit 1
  else
    cd irstlm
    # Just using the default aclocal, automake.
    # You may have to mess with the version by editing
    # regenerate-makefiles.sh if this does not work. 
    # We try regenerate-makefiles.sh twice as we have found that
    # under some circumstances this makes it work.
    ./regenerate-makefiles.sh || ./regenerate-makefiles.sh || exit 1;

   ./configure --prefix=`pwd` || exit 1

    # [ you may have to install zlib before typing make ]
    make || exit 1
    make install || exit 1
    cd ..
  fi
)
ok_irstlm=$?
if [ $ok_irstlm -ne 0 ]; then
  echo "****Installation of IRSTLM failed [not needed for most steps, anyway]."
fi


(
  # (5) Install sclite [OPTIONAL!]
  # This can be helpful helpful for scoring but the default scoring scripts do not
  # use it (they use our own Kaldi-based scorer).
  echo "**** (5) install sclite (optional; useful for detailed scoring output but the default scripts don't use it)"
  rm sctk-2.4.0-20091110-0958.tar.bz2  2>/dev/null
  wget -T 10 -t 3 ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.0-20091110-0958.tar.bz2

  if [ ! -e sctk-2.4.0-20091110-0958.tar.bz2 ]; then
    echo "download sctk-2.4.0-20091110-0958.tar.bz2 failed."
    exit 1
  else
    bunzip2 sctk-2.4.0-20091110-0958.tar.bz2 || exit 1
    gzip -f sctk-2.4.0-20091110-0958.tar || exit 1

    tar -xovzf sctk-2.4.0-20091110-0958.tar.gz  || exit 1
    cd sctk-2.4.0
    make config || exit 1
    make all || exit 1
    # Not doing the checks, they don't always succeed and it
    # it doesn't really matter.
    # make check || exit 1
    make install || exit 1
    make doc || exit 1
    cd ..
  fi
)
ok_sclite=$?
if [ $ok_sclite -ne 0 ]; then
  echo "****Installation of SCLITE failed [not needed anyway]."
fi

# (6) Install instructions for OpenFst

# Note that this should be compiled with g++-4.x
# You may have to install this and give the option CXX=<g++-4-binary-name>
# to configure, if it's not already the default (g++ -v will tell you).
# (on cygwin you may have to install the g++-4.0 package and give the options CXX=g++-4.exe CC=gcc-4.exe to configure).

(
  echo "****(6) Install openfst"

  rm openfst-1.2.7.tar.gz 2>/dev/null
  wget -T 10 -t 3 http://openfst.cs.nyu.edu/twiki/pub/FST/FstDownload/openfst-1.2.7.tar.gz

  if [ ! -e openfst-1.2.7.tar.gz ]; then
    echo "****download openfst-1.2.7.tar.gz failed."
    exit 1
  else
    tar -xovzf openfst-1.2.7.tar.gz   || exit 1
    cp partition.h minimize.h openfst-1.2.7/src/include/fst
    #ignore errors in the following; it's for robustness in case
    # someone follows these instructions after the installation of openfst.
    cp partition.h minimize.h openfst-1.2.7/include/fst 2>/dev/null
    # Remove any existing link
    rm openfst 2>/dev/null
    ln -s openfst-1.2.7 openfst
     
    cd openfst-1.2.7
    # Choose the correct configure statement:

    # Linux or Darwin:
    if [ "`uname`" == "Linux"  ] || [ "`uname`" == "Darwin"  ]; then
        ./configure --prefix=`pwd` --enable-static --disable-shared || exit 1
    elif [ "`uname -o`" == "Cygwin"  ]; then
        which gcc-4.exe || exit 1
        ./configure --prefix=`pwd` CXX=g++-4.exe CC=gcc-4.exe --enable-static --disable-shared  || exit 1
    else
        echo "Platform detection error"
        exit 1
    fi

    # make install is equivalent to "make; make install"
    make install || exit 1
    cd ..
  fi
)
ok_openfst=$?
if [ $ok_openfst -ne 0 ]; then
  echo "****Installation of OpenFst failed"
fi

echo
echo Install summary:

if [ $ok_sph2pipe -eq 0 ]; then
   echo "sph2pipe:Success"
else
   echo "sph2pipe:Failure"
fi
if [ $ok_atlas -eq 0 ]; then
   echo "ATLAS:   Success"
else
   echo "ATLAS:   Failure"
fi
if [ $ok_clapack -eq 0 ]; then
   echo "CLAPACK: Success"
else
   echo "CLAPACK: Failure"
fi
if [ $ok_irstlm -eq 0 ]; then
   echo "irstlm:  Success"
else
   echo "irstlm:  Failure [optional anyway]"
fi
if [ $ok_sclite -eq 0 ]; then
   echo "sclite:  Success"
else
   echo "sclite:  Failure [optional anyway]"
fi
if [ $ok_openfst -eq 0 ]; then
   echo "openfst: Success"
else
   echo "openfst: Failure"
fi


