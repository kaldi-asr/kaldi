#ifndef __HIPIFY_H__
#define __HIPIFY_H__

#ifdef __HIPCC__
inline __device__ void __syncwarp(unsigned mask = 0xffffffff) {
  // On CDNA hardware wave-fronts (warps) execute always in
  // lock step. Though it might still be important to signal
  // that the compiler can't reorder code around certain code
  // sections that rely on data sharing mecanisms like LDS
  // (shared memory). So this implements a No-op but is seen
  // by the compiler as having side effects.
  __asm__("s_nop 0");

  // A saffest option, arguably less performant would be to use:
  // __asm__("s_waitcnt lgkmcnt(0)"); Í
  // to explicitly do a memory fence.
}
// AMDGCN only support this rounding mode.
#define __fdiv_rd __fdiv_rn
#else
#define __align__(x) __attribute__((aligned(x)))
#endif

//
// HIP types
//
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F \
  HIPBLAS_R_32F  // TODO: Verify that plain float compute are viable
                 // replacements for the tensor cores alternative.
#define CUBLAS_COMPUTE_32F_FAST_TF32 \
  HIPBLAS_R_32F  // TODO: Verify that plain float compute are viable
                 // replacements for the tensor cores alternative.
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP \
  HIPBLAS_GEMM_DEFAULT  // TODO: Verify regular GEMMs are viable replacements
                        // for explicit tensor GEMMs.
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_R_32F HIPBLAS_R_32F
#define CUBLAS_R_64F HIPBLAS_R_64F
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_UNKNOWN
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define CURAND_STATUS_ALLOCATION_FAILED HIPRAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH HIPRAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED \
  HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_INITIALIZATION_FAILED HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_INITIALIZATION_FAILED HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_INTERNAL_ERROR HIPRAND_STATUS_INTERNAL_ERROR
#define CURAND_STATUS_LAUNCH_FAILURE HIPRAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_NOT_INITIALIZED HIPRAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_OUT_OF_RANGE HIPRAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_PREEXISTING_FAILURE HIPRAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_STATUS_TYPE_ERROR HIPRAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_VERSION_MISMATCH HIPRAND_STATUS_VERSION_MISMATCH
#define CUSPARSE_ACTION_NUMERIC HIPSPARSE_ACTION_NUMERIC
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_ORDER_COL HIPSPARSE_ORDER_COLUMN
#define CUSPARSE_SPMM_CSR_ALG2 HIPSPARSE_SPMM_CSR_ALG2
#define CUSPARSE_STATUS_ALLOC_FAILED HIPSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_ARCH_MISMATCH HIPSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_EXECUTION_FAILED HIPSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_INSUFFICIENT_RESOURCES \
  HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES
#define CUSPARSE_STATUS_INTERNAL_ERROR HIPSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_INVALID_VALUE HIPSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_MAPPING_ERROR HIPSPARSE_STATUS_MAPPING_ERROR
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED \
  HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_NOT_INITIALIZED HIPSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_NOT_SUPPORTED HIPSPARSE_STATUS_NOT_SUPPORTED
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_STATUS_ZERO_PIVOT HIPSPARSE_STATUS_ZERO_PIVOT
#define cuDeviceGetName hipDeviceGetName
#define cuMemGetInfo_v2 hipMemGetInfo
#define cublasComputeType_t hipblasDatatype_t
#define cublasCreate hipblasCreate
#define cublasDasum_v2 hipblasDasum
#define cublasDaxpy_v2 hipblasDaxpy
#define cublasDcopy_v2 hipblasDcopy
#define cublasDdot_v2 hipblasDdot
#define cublasDestroy hipblasDestroy
#define cublasDgemmBatched hipblasDgemmBatched
#define cublasDgemm_v2 hipblasDgemm
#define cublasDgemv_v2 hipblasDgemv
#define cublasDger_v2 hipblasDger
#define cublasDnrm2_v2 hipblasDnrm2
#define cublasDscal_v2 hipblasDscal
#define cublasDspmv_v2 hipblasDspmv
#define cublasDspr_v2 hipblasDspr
#define cublasDsyrk_v2 hipblasDsyrk
#define cublasDtpmv_v2 hipblasDtpmv
#define cublasDtrsm_v2(a, b, c, d, e, f, g, h, i, j, k, l) \
  hipblasDtrsm(a, b, c, d, e, f, g, h, const_cast<double*>(i), j, k, l)
#define cublasFillMode_t hipblasFillMode_t
#define cublasGemmAlgo_t hipblasGemmAlgo_t
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasSasum_v2 hipblasSasum
#define cublasSaxpy_v2 hipblasSaxpy
#define cublasScopy_v2 hipblasScopy
#define cublasSdot_v2 hipblasSdot
#define cublasSetStream hipblasSetStream
#define cublasSgemv_v2 hipblasSgemv
#define cublasSger_v2 hipblasSger
#define cublasSnrm2_v2 hipblasSnrm2
#define cublasSscal_v2 hipblasSscal
#define cublasSspmv_v2 hipblasSspmv
#define cublasSspr_v2 hipblasSspr
#define cublasSsyrk_v2 hipblasSsyrk
#define cublasStatus_t hipblasStatus_t
#define cublasStatus_t hipblasStatus_t
#define cublasStpmv_v2 hipblasStpmv
#define cublasStrsm_v2(a, b, c, d, e, f, g, h, i, j, k, l) \
  hipblasStrsm(a, b, c, d, e, f, g, h, const_cast<float*>(i), j, k, l)
#define cudaComputeModeExclusive hipComputeModeExclusive
#define cudaComputeModeExclusiveProcess hipComputeModeExclusiveProcess
#define cudaDataType hipDataType
#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorDeviceAlreadyInUse hipErrorContextAlreadyInUse
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocPitch hipMallocPitch
#define cudaMemcpy hipMemcpy
// hipMemcpy2DAsync has a disparity to its CUDA counterpart for zero-sized
// copies, which should be canceled by ROCm 5.7.1+. Then the following would
// be sufficient:
// #define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpy2DAsync(a, b, c, d, width, height, e, f)      \
  [&]() -> hipError_t {                                         \
    if (width && height)                                        \
      return hipMemcpy2DAsync(a, b, c, d, width, height, e, f); \
    return hipSuccess;                                          \
  }()
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaProfilerStop hipProfilerStop
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread ((hipStream_t)2)
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cufftComplex hipfftComplex
#define cufftDestroy hipfftDestroy
#define cufftExecR2C hipfftExecR2C
#define cufftHandle hipfftHandle
#define cufftPlanMany hipfftPlanMany
#define cufftSetStream hipfftSetStream
#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerator_t hiprandGenerator_t
#define curandSetGeneratorOffset hiprandSetGeneratorOffset
#define curandSetGeneratorOrdering(x, y) \
  0  // HIP does not support generator ordeing.
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandSetStream hiprandSetStream
#define curandStatus_t hiprandStatus_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverDnSetStream hipsolverDnSetStream
#define cusolverDnSpotrf hipsolverDnSpotrf
#define cusolverDnSpotrfBatched hipsolverDnSpotrfBatched
#define cusolverDnSpotrf_bufferSize hipsolverDnSpotrf_bufferSize
#define cusolverDnSpotrs hipsolverDnSpotrs
#define cusolverDnSpotrsBatched hipsolverDnSpotrsBatched
#define cusparseAction_t hipsparseAction_t
#define cusparseCreate hipsparseCreate
#define cusparseCreateCsr hipsparseCreateCsr
#define cusparseCreateDnMat hipsparseCreateDnMat
#define cusparseCreateMatDescr hipsparseCreateMatDescr
#define cusparseDcsr2csc hipsparseDcsr2csc
#define cusparseDestroy hipsparseDestroy
#define cusparseDestroy hipsparseDestroy
#define cusparseDestroyDnMat hipsparseDestroyDnMat
#define cusparseDestroyMatDescr hipsparseDestroyMatDescr
#define cusparseDestroySpMat hipsparseDestroySpMat
#define cusparseDnMatDescr_t hipsparseDnMatDescr_t
#define cusparseGetMatIndexBase hipsparseGetMatIndexBase
#define cusparseHandle_t hipsparseHandle_t
#define cusparseIndexBase_t hipsparseIndexBase_t
#define cusparseMatDescr_t hipsparseMatDescr_t
#define cusparseOperation_t hipsparseOperation_t
#define cusparseScsr2csc hipsparseScsr2csc
#define cusparseSetStream hipsparseSetStream
#define cusparseSpMM hipsparseSpMM
#define cusparseSpMM_bufferSize hipsparseSpMM_bufferSize
#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#define cusparseStatus_t hipsparseStatus_t
#define nvtxRangePop roctxRangePop
#define nvtxRangePush roctxRangePush
#define nvtxRangePushA roctxRangePushA
//
// HIPCUB namespace.
//
#define cub hipcub

//
// Callback qualifier
//
#define CUDART_CB

//
// Math constants
//
#define CUDART_INF HIP_INF
#define CUDART_INF_F HIP_INF_F

//
// GPU static hardware characteristics.
//
#define GPU_WARP_SIZE 64
#define GPU_MAX_THREADS_PER_BLOCK 1024
#define GPU_MAX_WARPS_PER_BLOCK (GPU_MAX_THREADS_PER_BLOCK / GPU_WARP_SIZE)
#endif  //__HIPIFY_H__
