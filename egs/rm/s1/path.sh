
root=`pwd`/../../..
#valgrind=yes
valgrind=no

# If you run with valgrind, by setting valgrind=yes above, 
# the errors can be seen with the following command:
# ( grep 'ERROR SUMMARY' exp/*/*.log | grep -v '0 errors' ;  grep 'definitely lost' exp/*/*.log | grep -v -w 0 )

if [ $valgrind == "no" ]; then
  export PATH=${root}/src/bin:${root}/tools/openfst/bin:${root}/src/fstbin/:${root}/src/gmmbin/:${root}/src/featbin/:${root}/src/fgmmbin:${root}/src/sgmmbin:${root}/src/lm:${root}/src/latbin:$PATH  
else 
  mkdir bin
  for x in ${root}/src/{bin,fstbin,gmmbin,featbin,fgmmbin,sgmmbin,lm,latbin}; do
    for y in $x/*; do
      if [ -x $y ]; then
        b=`basename $y`
        echo valgrind $y '"$@"' > bin/$b
        chmod +x bin/`basename $b`
      fi
    done
  done
  export PATH=`pwd`/bin/:${root}/tools/openfst/bin:$PATH
fi

export LC_ALL=C
export LC_LOCALE_ALL=C

