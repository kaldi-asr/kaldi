#include <iostream>
#include <stdio.h>
#include <cuda.h>

#ifdef __CUDACC__
  #define HOST __host__
  #define DEVICE __device__

#else
  #define HOST
  #define DEVICE
#endif

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


DEVICE void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

DEVICE void release_semaphore(volatile int *lock){
  //*lock = 0;
  atomicExch((unsigned int*)lock,0u);
  __threadfence();
  }

  template<int blockDimx, int blockDimy>
  inline DEVICE void myadd(int *ret, volatile int *mutex) {
    acquire_semaphore((int*)(mutex+threadIdx.x));
    (*(ret+threadIdx.x))++;
    release_semaphore((int*)(mutex+threadIdx.x));
  }

  template<int blockDimx, int blockDimy>
  inline DEVICE void myadd2(int *ret, volatile int *mutex) {
    if (threadIdx.x==0) {
    acquire_semaphore((int*)(mutex+threadIdx.x));
    (*(ret+threadIdx.x))++;
    release_semaphore((int*)(mutex+threadIdx.x));
    }
  }
  template<int blockDimx, int blockDimy>
  inline DEVICE void myadd0(int *ret, volatile int *mutex) {
    acquire_semaphore((int*)(mutex));
    (*(ret))++;
    release_semaphore((int*)(mutex));
  }

  __global__ void callmyadd(int *ret, int *mutex) {
  //myadd2<32,2>(ret, mutex);
  myadd0<32,2>(ret, mutex);
  //myadd<32,2>(ret, mutex);
  }
int main() {
  //int blocks=200;
  int blocks=3;
  //int blocks=7;
  int *mutex=0;
  int *ret=0, ret_h=0;
  int n =32;

  cudaMalloc((void**) &mutex, n*sizeof(int));  
  cudaMalloc((void**) &ret, n*sizeof(int));
  cudaMemset(mutex, 0,n*sizeof(int));
  cudaMemset(ret, 0,n*sizeof(int));

  callmyadd<<<blocks,n>>>(ret, mutex);
  cudaCheckError();
  cudaMemcpy(&ret_h, ret, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << ret_h <<std::endl;
    
  cudaFree(ret);
  cudaFree(mutex);
}
