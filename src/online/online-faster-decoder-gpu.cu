#include "stdio.h"
#include "iostream"
#include "stdlib.h"

typedef int int32;
typedef long long int64;

struct GPUToken{
    typedef float BaseFloat;
    
    GPUToken* prev_best_tok;
    BaseFloat cost;
};

struct GPUArc{ 
    typedef int StateId;

    StateId curstate;
    Arc arc;
};

__global__ ProcessEmitting();
__global__ ProcessNonEmitting();

//ini gimana cara bikin supaya DecodableInterface diassign di CUDA
//EDIT : Pake kelasnya langsung aja, jadi pake OnlineDecodableDiagGmmScaled langsung
__global__ Decode(OnlineDecodableDiagGmmScaled* decodable);
