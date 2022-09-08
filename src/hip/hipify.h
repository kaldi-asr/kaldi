#ifndef __HIPIFY_H__
#define __HIPIFY_H__

inline __device__ void __syncwarp(unsigned mask=0xffffffff) {}

//
// HIP types
// TODO: Verify that HIPBLAS_R_32F and HIPBLAS_GEMM_DEFAULT can be sensible replacements for tensor ops.
//

#define cudaDevAttrWarpSize     hipDeviceAttributeWarpSize
#define cudaDeviceGetAttribute  hipDeviceGetAttribute
#define cudaGetDevice           hipGetDevice
#define cudaGetErrorString      hipGetErrorString
#define cudaStream_t            hipStream_t
#define cudaStreamLegacy        ((hipStream_t)1)
#define cudaStreamPerThread     ((hipStream_t)2)
#define cublasStatus_t          hipblasStatus_t
#define cudaError_t             hipError_t
#define cusparseDestroy         hipsparseDestroy
#define cudaGetLastError        hipGetLastError

#define cudaFree  hipFree
#define cudaGetErrorString hipGetErrorString
#define cublasCreate hipblasCreate
#define cublasSetStream hipblasSetStream
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define curandCreateGenerator hiprandCreateGenerator
#define curandSetStream hiprandSetStream
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDeviceProperties hipGetDeviceProperties
#define curandDestroyGenerator hiprandDestroyGenerator
#define cusparseDestroy hipsparseDestroy
#define cudaDeviceProp hipDeviceProp_t
#define cublasOperation_t hipblasOperation_t
#define cublasStatus_t hipblasStatus_t
#define cusparseStatus_t hipsparseStatus_t
#define curandStatus_t hiprandStatus_t
#define cublasHandle_t  hipblasHandle_t
#define cusparseHandle_t hipsparseHandle_t
#define curandGenerator_t hiprandGenerator_t
#define cublasGemmAlgo_t hipblasGemmAlgo_t
#define cusolverDnHandle_t  hipsolverDnHandle_t
#define cublasComputeType_t hipblasDatatype_t
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandSetGeneratorOffset hiprandSetGeneratorOffset
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cublasDaxpy_v2 hipblasDaxpy
#define cublasSaxpy_v2 hipblasSaxpy
#define cublasDscal_v2 hipblasDscal
#define cublasSscal_v2 hipblasSscal
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnSetStream hipsolverDnSetStream
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define cusparseCreate hipsparseCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusparseSetStream hipsparseSetStream
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define curandSetGeneratorOrdering(x,y) 0 // HIP does not support generator ordeing.
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceReset hipDeviceReset
#define cudaComputeModeExclusive hipComputeModeExclusive
#define cudaComputeModeExclusiveProcess hipComputeModeExclusiveProcess
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cublasDestroy hipblasDestroy
#define cuDeviceGetName hipDeviceGetName
#define cudaErrorDeviceAlreadyInUse hipErrorContextAlreadyInUse
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#define cusparseMatDescr_t hipsparseMatDescr_t
#define cudaMemsetAsync hipMemsetAsync
#define cublasGemmEx hipblasGemmEx
#define cublasDgemm_v2 hipblasDgemm
#define cublasSger_v2 hipblasSger
#define cublasDger_v2 hipblasDger
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasDgemmBatched hipblasDgemmBatched
#define cublasStrsm_v2(a,b,c,d,e,f,g,h,i,j,k,l) hipblasStrsm(a,b,c,d,e,f,g,h,const_cast<float*>(i),j,k,l)
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define cublasDtrsm_v2(a,b,c,d,e,f,g,h,i,j,k,l) hipblasDtrsm(a,b,c,d,e,f,g,h,const_cast<double*>(i),j,k,l)
#define cublasFillMode_t hipblasFillMode_t
#define cublasSsyrk_v2 hipblasSsyrk
#define cublasDsyrk_v2 hipblasDsyrk
#define cublasSdot_v2 hipblasSdot
#define cublasSasum_v2 hipblasSasum
#define cublasDnrm2_v2 hipblasDnrm2
#define cublasScopy_v2 hipblasScopy
#define cublasDcopy_v2 hipblasDcopy
#define cublasSgemv_v2 hipblasSgemv
#define cublasDgemv_v2 hipblasDgemv
#define cublasSspmv_v2 hipblasSspmv
#define cublasDspmv_v2 hipblasDspmv
#define cublasDtpmv_v2 hipblasDtpmv
#define cublasSspr_v2 hipblasSspr
#define cublasDspr_v2 hipblasDspr
#define cudaDataType hipDataType
#define cusparseAction_t hipsparseAction_t
#define cublasDdot_v2 hipblasDdot
#define cublasDasum_v2 hipblasDasum
#define cublasSnrm2_v2 hipblasSnrm2
#define cublasStpmv_v2 hipblasStpmv
#define cusparseIndexBase_t hipsparseIndexBase_t
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define cusparseOperation_t hipsparseOperation_t
#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#define cusparseGetMatIndexBase hipsparseGetMatIndexBase
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define cusparseCreateCsr hipsparseCreateCsr
#define cusparseDnMatDescr_t hipsparseDnMatDescr_t
#define CUSPARSE_ORDER_COL HIPSPARSE_ORDER_COLUMN
#define cusparseCreateDnMat hipsparseCreateDnMat
#define CUSPARSE_SPMM_CSR_ALG2 HIPSPARSE_SPMM_CSR_ALG2
#define cusparseSpMM_bufferSize hipsparseSpMM_bufferSize
#define cusparseSpMM hipsparseSpMM
#define cusparseDestroySpMat hipsparseDestroySpMat
#define cusparseDestroyDnMat hipsparseDestroyDnMat
#define cusparseScsr2csc hipsparseScsr2csc
#define CUDA_R_64F HIP_R_64F
#define CUDA_R_32F HIP_R_32F
#define CUBLAS_R_64F HIPBLAS_R_64F
#define CUBLAS_R_32F HIPBLAS_R_32F
#define cusparseDcsr2csc hipsparseDcsr2csc
#define cusparseCreateMatDescr hipsparseCreateMatDescr
#define cusparseDestroyMatDescr hipsparseDestroyMatDescr
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_N HIPBLAS_OP_N
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset2DAsync hipMemset2DAsync
//
// HIPCUB
//
#define cub hipcub


#endif //__HIPIFY_H__
