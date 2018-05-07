#define GPUFST_GPUFST_GPU_UTILS_HPP

#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gpufst{

template <int I, typename T, class... Types>
void unzip_to_device(const std::vector<std::tuple<Types...>> &src, thrust::device_vector<T> &dst) {
  cudaMemcpy2D(dst.data().get(), sizeof(T),
	       &std::get<I>(src[0]),  sizeof(std::tuple<Types...>),
	       sizeof(typename std::tuple_element<I, std::tuple<Types...>>::type), 
	       src.size(), cudaMemcpyHostToDevice);
}

__device__ void AtomicAdd2(float *address, float value){
   int oldval, newval, readback;
 
   oldval = __float_as_int(*address);
   newval = __float_as_int(__int_as_float(oldval) + value);
   while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) 
     {
      oldval = readback;
      newval = __float_as_int(__int_as_float(oldval) + value);
     }
}
__device__ void AtomicAdd(float * const address, const float value)
{

        int * const address_as_i = (int *)address;
        int old = * address_as_i, assumed;

        do
        {
                assumed = old;
                if (__int_as_float(assumed) >= value)
                {
                        break;
                }

                old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        } while (assumed != old);
}

__device__ void AtomicMax(float * const address, const float value)
{
        if (* address >= value)
        {
                return;
        }

        int * const address_as_i = (int *)address;
        int old = * address_as_i, assumed;

        do
        {
                assumed = old;
                if (__int_as_float(assumed) >= value)
                {
                        break;
                }

                old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        } while (assumed != old);
}

__device__ void AtomicMin(float * const address, const float value)
{
        if (* address <= value)
        {
                return;
        }

        int * const address_as_i = (int *)address;
        int old = * address_as_i, assumed;

        do
        {
                assumed = old;
                if (__int_as_float(assumed) <= value)
                {
                        break;
                }

                old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        } while (assumed != old);
}


__device__ void atomicLogAdd(float * const address, const float val) {
	signed int* address_as_ull = (signed int*)address;
        signed int old = *address_as_ull, assumed;

        do {
                assumed = old;
                if(val >= __int_as_float(assumed)){
                        old = atomicCAS(address_as_ull, assumed, __float_as_int(val+ log1pf(expf(__int_as_float(assumed)-val))));
                }
                else{
                        old = atomicCAS(address_as_ull, assumed, __float_as_int(__int_as_float(assumed)+ log1pf(expf(val-__int_as_float(assumed)))));
                }
        }
        while (assumed != old);
}

//x = backward[from_states[foo]]
//y = (backward_prev[to_states[foo]] + probs[foo])
__device__ void logAdd(float * const x,const float y) {
	    //signed int* address_as_ull = (signed int*)x;
	    //signed int old = *address_as_ull;

	    float val = *x;
            if (y >= val ){
                *x = y+log1pf(expf(val-y));
		//atomicCAS(address_as_ull,assumed,__float_as_int(y+log1pf(expf(*x-y))));
            }
            else{
                *x = val + log1pf(expf(y-val));
		//atomicCAS(address_as_ull,assumed,__float_as_int(*x+log1pf(expf(y-*x))));
            }
}

__device__ void atomicLogAdd_iter(float * const address, const float val,int *iterations) {
        signed int* address_as_ull = (signed int*)address;
        signed int old = *address_as_ull, assumed;

        do {
                atomicAdd(&iterations[0],1);
                assumed = old;
                if(val >= __int_as_float(assumed)){
                        old = atomicCAS(address_as_ull, assumed, __float_as_int(val+ log1pf(expf(__int_as_float(assumed)-val))));
                }
                else{
                        old = atomicCAS(address_as_ull, assumed, __float_as_int(__int_as_float(assumed)+ log1pf(expf(val-__int_as_float(assumed)))));
                }
        }
        while (assumed != old);
}




// For an explanation about this method read: 
//https://gasstationwithoutpumps.wordpress.com/2014/05/06/sum-of-probabilities-in-log-prob-space/
__device__ float atomicLogAdd2(float* address, float val) {
        signed int* address_as_ull = (signed int*)address;
        signed int old = *address_as_ull, assumed;
	
        do {
                assumed = old;
		if(val >= __int_as_float(assumed)){
                        old = atomicCAS(address_as_ull, assumed, __float_as_int(val+ logf(expf(__int_as_float(assumed)-val)+1)));
                }
                else{
			old = atomicCAS(address_as_ull, assumed, __float_as_int(__int_as_float(assumed)+ logf(expf(val-__int_as_float(assumed))+1)));
                }
        }
        while (assumed != old);
        return __int_as_float(old);
}

// Method to perform
__device__ void atomicAdd_exp_man(float* xm, int* xe,int ye,float val,float val2) {
        signed int* address_as_ull = (signed int*)xm;
	signed int* address_as_ull2 = (signed int*)xe;
        signed int old = *address_as_ull, assumed;
	signed int old2 = *address_as_ull2, assumed2;

        do {
                assumed = old;
		assumed2 = old2;

                if(ye > __int_as_float(assumed2)){
                        old = atomicCAS(address_as_ull, assumed, __float_as_int(val));
			old2 = atomicCAS(address_as_ull2, assumed2, __float_as_int(ye));
			old = atomicCAS(address_as_ull, assumed, __float_as_int(__int_as_float(assumed)+val2));
                }
                else{
			old = atomicCAS(address_as_ull, assumed, __float_as_int(__int_as_float(assumed)+val2));
                }
        }
        while (assumed != old);
}

}