# make clean
rm -f *.o rnnlm.cpu
g++ -g -O2 cued-main.cc cued-rnnlm-lib.cc Mathops.cc fileops.cc helper.cc  -o rnnlm.cpu -lrt -fopenmp

make
