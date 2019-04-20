#!/bin/bash
export LDFLAGS="-L`pwd`/openfst/lib"
export CXXFLAGS="-I`pwd`/openfst/include"
stage=0

if [ $stage -le 0 ] ; then
    rm -rf re2 protobuf sparrowhawk*
    git clone -b feature/Spanish_normalizer https://github.com/spokencloud/sparrowhawk-resources.git || exit 1;
    patch -p0 < sparrowhawk-resources/local/Makefile.patch || exit 1;
    make openfst || exit 1;
    git clone https://github.com/mjansche/thrax.git
    export LDFLAGS=-L`pwd`/openfst/lib
    export CXXFLAGS=-I`pwd`/openfst/include
    cd thrax
    autoreconf --force --install || exit 1;
    ./configure --prefix=`pwd` || exit 1;
    make || exit 1;
    make install || exit 1;
    cd ..
    git clone https://github.com/google/re2.git || exit 1;
    cd re2/
    make -j 20 || exit 1;
    make test || exit 1;
    make install prefix=`pwd` || exit 1;
    cd ..
    git clone https://github.com/google/protobuf.git || exit 1;
    cd protobuf/
    ./autogen.sh || exit 1;
    ./configure --prefix=`pwd` || exit 1;
    make -j 20 || exit 1;
    make install || exit 1;
    cd ..
fi

if [ $stage -le 1 ]; then 
    git clone https://github.com/google/sparrowhawk.git || exit 1;
    patch -p0 < sparrowhawk-resources/local/sparrowhawk.patch || exit 1;
    cd sparrowhawk/ || exit 1;
    mkdir lib
    mkdir bin
    mkdir include
    cp -r ../openfst/lib/* lib/ || exit 1;
    cp -r ../protobuf/lib/* lib/ || exit 1;
    cp -r ../re2/lib/* lib/ || exit 1;
    cp -r ../thrax/lib/* lib/ || exit 1;
    cp -r ../openfst/include/* include/ || exit 1;
    cp -r ../protobuf/include/* include/ || exit 1;
    cp -r ../re2/include/* include/ || exit 1;
    cp -r ../thrax/include/* include/ || exit 1;
    cp ../protobuf/bin/protoc bin/. || exit 1;
    export PATH=`pwd`/bin:$PATH
    aclocal || exit 1;
    automake || exit 1;
    ./configure --prefix=`pwd`  CPPFLAGS="-I`pwd`/include"  LDFLAGS="-L`pwd`/lib" || exit 1;
    make || exit 1;
    make install || exit 1;
    cd ..
fi

if [ $stage -le 2 ]; then 
    cp -r sparrowhawk-resources/language-resources sparrowhawk/ || exit 1;
    cd sparrowhawk/language-resources/esp/textnorm/classifier || exit 1;
    . ./path.sh || exit 1;
    python2 create_far.py ascii.syms  universal_depot_ascii universal_depot universal_depot.far 
    thraxmakedep tokenize_and_classify.grm || exit 1;
    make || exit 1;
    cd ../verbalizer
    python2 create_far.py ascii.syms  number_names_depot_ascii number_names_depot number_names_depot.far
    cp -r ../classifier/universal_depot.far .
    thraxmakedep  verbalize.grm || exit 1;
    make || exit 1;
    cd ../../../../..
fi
