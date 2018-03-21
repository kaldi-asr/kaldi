#include "stdio.h"
#include "iostream"
#include "stdlib.h"

__global__ ProcessEmitting();
__global__ ProcessNonEmitting();

__global__ Decode(DecodableInterface* decodable) //ini gimana cara bikin supaya DecodableInterface diassign di CUDA