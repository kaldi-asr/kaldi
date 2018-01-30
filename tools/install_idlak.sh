#!/bin/bash
# This script attempts to automatically execute the instructions in 
# INSTALL_IDLAK.

# (1) Install instructions for expat

if ! which wget >&/dev/null; then
   echo "This script requires you to first install wget";
   exit 1;
fi

if ! which cmake >&/dev/null; then
  echo "This script requires cmake: install it first.";
  exit 1;
fi

echo "****(1) Installing expat"

(
  rm -f expat-2.1.0.tar.gz 2>/dev/null
  wget -T 10 -t 3 https://sourceforge.net/projects/expat/files/expat/2.1.0/expat-2.1.0.tar.gz
  if [ ! -e expat-2.1.0.tar.gz ]; then
    echo "****download of expat-2.1.0.tar.gz failed."
    exit 1
  else
    tar -xovzf expat-2.1.0.tar.gz || exit 1
    cd expat-2.1.0
    ./configure --prefix=`pwd` || exit 1
    make || exit 1
    make install || exit 1
    cd ..
  fi
)
ok_expat=$?
if [ $ok_expat -ne 0 ]; then
  echo "****expat install failed."
  exit 1
fi

echo "****(2) Installing pugixml"

(
  rm -rf pugixml-1.2 pugixml-1.2.tar.gz 2>/dev/null
  #wget -T 10 -t 3 http://pugixml.googlecode.com/files/pugixml-1.2.tar.gz
  wget -T 10 -t 3 https://github.com/zeux/pugixml/releases/download/v1.2/pugixml-1.2.tar.gz
  if [ ! -e pugixml-1.2.tar.gz ]; then
    echo "****download of pugixml-1.2.tar.gz failed."
    exit 1
  else
    mkdir pugixml-1.2
    cd pugixml-1.2
    tar -xovzf ../pugixml-1.2.tar.gz || exit 1
    cd scripts
    if [ "`uname`" == "Darwin"  ]; then
      # OS X 10.9, 10.10 require CXXFLAGS += -stdlib=libstdc++ to compile pugixml
      osx_ver=`sw_vers | grep ProductVersion | awk '{print $2}' | awk '{split($0,a,"\."); print a[1] "." a[2]; }'`
      echo "Configuring for OS X version $osx_ver ..."
      if [ "$osx_ver" == "10.9" ]; then
        cmake -DCMAKE_CXX_FLAGS=-stdlib=libstdc++
      elif [ "$osx_ver" == "10.10" ]; then
        cmake -DCMAKE_CXX_FLAGS=-stdlib=libstdc++
      else
        cmake .
      fi
    else
      cmake . || exit 1
    fi
    make || exit 1
    cd ../..
  fi
)
ok_pugixml=$?
if [ $ok_pugixml -ne 0 ]; then
  echo "****pugixml install failed."
  exit 1
fi

echo "****(3) Installing pcre with utf8 support"

(
  rm -f pcre-8.20.tar.bz2 2>/dev/null
  wget -T 10 -t 3 https://sourceforge.net/projects/pcre/files/pcre/8.20/pcre-8.20.tar.bz2
  if [ ! -e pcre-8.20.tar.bz2 ]; then
    echo "****download of pcre-8.20.tar.bz2 failed."
    exit 1
  else
    tar -xovjf pcre-8.20.tar.bz2 || exit 1
    cd pcre-8.20
    ./configure --enable-utf8 --enable-unicode-properties --enable-newline-is-anycrlf --prefix=`pwd` || exit 1
    make || exit 1
    make install || exit 1
    cd ..
  fi
)
ok_pcre=$?
if [ $ok_pcre -ne 0 ]; then
  echo "****pcre install failed."
  exit 1
fi

echo "****(4) Installing SPTK"
(
    rm -f SPTK-3.9.tar.gz 2>/dev/null
    wget -T 10 -t 3 https://sourceforge.net/projects/sp-tk/files/SPTK/SPTK-3.9/SPTK-3.9.tar.gz
    if [ ! -e SPTK-3.9.tar.gz ]; then
	echo "****download of SPTK-3.9.tar.gz failed."
	exit 1
    else
	mkdir -p SPTK
	tar -xovzf SPTK-3.9.tar.gz || exit 1
	cd SPTK-3.9
	./configure --prefix=`pwd`/../SPTK || exit 1
	make || exit 1
	make install || exit 1
    cd ..
  fi
)
ok_sptk=$?
if [ $ok_sptk -ne 0 ]; then
  echo "****sptk install failed."
  exit 1
fi

echo "****(5) Installing phonetisaurus"
#(
#    rm -f Phonetisaurus
#    git clone https://github.com/AdolfVonKleist/Phonetisaurus.git
#    if [ ! -e Phonetisaurus ]; then
#        echo "****cloning of Phonetisaurus failed."
#	    exit 1
#    else
#        cd Phonetisaurus
#	    git checkout 09651ed5f6e9040d6dd30070601ecccfad254df4 . || exit 1
#        patch -p1 -N < ../extras/phonetisaurus.patch
#        cd src/.autoconf
#        autoconf -o ../configure || exit 1
#        cd ..
#        LDFLAGS="-Wl,-rpath=`pwd`/../../openfst/lib/" ./configure --with-openfst-libs=`pwd`/../../openfst/lib --with-openfst-includes=`pwd`/../../openfst/include  --with-install-bin=`pwd`/.. || exit 1
#	    make -j4 || exit 1
#	    make install || exit 1
#        cd ..
#    fi
#)
#ok_phonetisaurus=$?
#if [ $ok_phonetisaurus -ne 0 ]; then
#    echo "****phonetisaurus install failed."
#    exit 1
#fi

# echo "****(1) Installing Apache Xerces C++ XML Parser"

# (
#   rm xerces-c-3.1.1.tar.gz 2>/dev/null
#   wget -T 10 -t 3 http://mirror.rmg.io/apache//xerces/c/3/sources/xerces-c-3.1.1.tar.gz
#   if [ ! -e xerces-c-3.1.1.tar.gz]; then
#     echo "****download of xerces-c-3.1.1.tar.gz failed."
#     exit 1
#   else
#     tar -xovzf xerces-c-3.1.1.tar.gz || exit 1
#     cd xerces-c-3.1.1
#     ./configure --prefix=`pwd` || exit 1
#     make || exit 1
#     make install || exit 1
#     cd ..
#   fi
# )
# ok_xerces=$?
# if [ $ok_xerces -ne 0 ]; then
#   echo "**** Apache Xerces C++ XML Parser install failed."
# fi

# echo "****(1) Installing libXML"

# (
#   rm libxml2-2.8.0.tar.gz 2>/dev/null
#   wget -T 10 -t 3 ftp://xmlsoft.org/libxml2/libxml2-2.8.0.tar.gz
#   if [ ! -e libxml2-2.8.0.tar.gz]; then
#     echo "****download of libxml2-2.8.0.tar.gz failed."
#     exit 1
#   else
#     tar -xovzf libxml2-2.8.0.tar.gz || exit 1
#     cd libxml2-2.8.0
#     ./configure --prefix=`pwd` || exit 1
#     make || exit 1
#     make install || exit 1
#     cd ..
#   fi
# )
# ok_libxml=$?
# if [ $ok_libxml -ne 0 ]; then
#   echo "**** libXML install failed."
# fi

# echo "****(2) Installing Arabica"

# (
#   rm arabica-2010-November.tar.bz2 2>/dev/null
#   wget -T 10 -t 3 http://sourceforge.net/projects/arabica/files/latest/download?source=files
#   if [ ! -e arabica-2010-November.tar.bz2]; then
#     echo "****download of arabica-2010-November.tar.bz2 failed."
#     exit 1
#   else
#     tar -xovjf arabica-2010-November.tar.bz2 || exit 1
#     cd libxml2-2.8.0
#     ./configure --prefix=`pwd` --with-libxml2=`pwd`/../libxml2-2.8.0/lib || exit 1
#     make || exit 1
#     make install || exit 1
#     cd ..
#   fi
# )
# ok_xerces=$?
# if [ $ok_xerces -ne 0 ]; then
#   echo "**** Apache Xerces C++ XML Parser install failed."
# fi
