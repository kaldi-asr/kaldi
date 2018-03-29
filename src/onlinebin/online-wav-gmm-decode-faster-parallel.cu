#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "online/online-faster-decoder-parallel.cuh"

int main(void){

  try{
    using namespace kaldi;
    using namespace fst;
  }
  catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}