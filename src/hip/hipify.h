#ifndef __HIPIFY_H__
#define __HIPIFY_H__

inline __device__ void __syncwarp(unsigned mask=0xffffffff) {}

//
// HIP types
//
#define cudaDevAttrWarpSize     hipDeviceAttributeWarpSize
#define cudaDeviceGetAttribute  hipDeviceGetAttribute
#define cudaGetDevice           hipGetDevice
#define cudaStream_t            hipStream_t
#define cudaStreamLegacy        ((hipStream_t)1)
#define cudaStreamPerThread     ((hipStream_t)2)

//
// HIPCUB
//
#define cub hipcub


#endif //__HIPIFY_H__
