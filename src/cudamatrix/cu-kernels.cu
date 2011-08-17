
#include <cfloat>
#include "cu-kernels.h"



/*
 * CUDA kernels
 */


/*
 * CuMatrix
 */
template<typename T>
__global__
static void _set_const(T* mat, T value, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = value;
}


template<typename T>
__global__
static void _apply_log(T* mat, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    if(mat[index] < FLT_MIN) mat[index] = FLT_MIN;
    mat[index] = log(mat[index]);
}


template<typename T>
__global__
static void _apply_mask(T* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*dmat.stride;
  int index2 = i + j*dmask.stride;
  if ( i < dmat.cols  &&  j < dmat.rows ) 
    if(mask[index2] == 0) mat[index] = 0;
}


template<typename T>
__global__
static void _regularize_l1(T* wei, T* grad, T l1, T lr, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows ) {

    if(wei[index]==0.0) return; //skip L1 if zero weight!
    
    T l1_signed = l1;
    if(wei[index] < 0.0) //flip sign
      l1_signed = -l1;

    T before = wei[index];
    T after = wei[index] -lr*grad[index] -l1_signed;//simulate update
    if((after > 0.0) ^ (before > 0.0)) { //sign changed?
      wei[index] = 0.0;
      grad[index] = 0.0;
    } else {
      wei[index] -= l1_signed;
    }
  }
}


template<typename T>
__global__
static void _scale_cols(T* mat, const T* scale, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] *= scale[i];
}


template<typename T>
__global__
static void _scale_rows(T* mat, const T* scale, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] *= scale[j];
}


template<typename T>
__global__
static void _add_scaled(T alpha, const T* A, T beta, T* dst, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    dst[index] = alpha*A[index] + beta*dst[index];
}


template<typename T>
__global__
static void _add_scaled_row(T alpha, const T* row, T beta, T* dst, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;

#if 0
  //this does not accelerate :(
  __shared__ T aux[16];
  if(threadIdx.y == 0 && i < d.cols) aux[threadIdx.x] = row[i];
  __syncthreads();
  
  if ( i < d.cols  &&  j < d.rows )
    dst[index] = alpha*aux[threadIdx.x] + beta*dst[index];
#else
  if ( i < d.cols  &&  j < d.rows )
    dst[index] = alpha*row[i] + beta*dst[index];
#endif
}


template<typename T>
__global__
static void _mul_elem(T* mat, const T* A, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = mat[index] * A[index];
}





/*
 * CuVector
 */
template<typename T>
__global__
static void _add_col_sum(T alpha, const T* mat, T beta, T* vec, MatrixDim d) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //This should be called 1-D
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(j > 0) return;
  
  if(i < d.cols) {
    double sum = 0.0;
    for(int k = 0; k < d.rows; k++) {
      sum += mat[i+k*d.stride];
    }
    vec[i] = alpha*sum + beta*vec[i];
  }
}


template<typename T>
__global__
static void _add_col_sum_reduce(T alpha, const T* mat, T beta, T* vec, MatrixDim d) {

  //flipped x,y for reducing... x..row, y..col
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if(blockIdx.x > 0) return;
  if(blockDim.y != 1) return;

  //copy vector to shared mem
  __shared__ T aux[512];
  aux[threadIdx.x] = mat[i+j*d.stride];
  __syncthreads();

  T sum = _sum_reduce(aux);
  __syncthreads();
  //copy out the result
  vec[i] = alpha*sum + beta*vec[i];
}



/*
 * cu::
 */
template<typename T>
__global__
static void _sigmoid(T*y, const T*x, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if( i < d.cols  &&  j < d.rows ) {
    T res = 1.0 / (1.0 + exp(-x[index]));
    /*
    if(res < 0.001) res = 0.001;
    if(res > 0.999) res = 0.999;
    */
    y[index] = res;
  }
}


template<typename T>
__global__
static void _diff_sigmoid(T*eout, const T*e, const T*y, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) 
    eout[index] = y[index]*(1.0-y[index]) * e[index];
}


template<typename T>
__global__
static void _softmax(T*y, const T*x, MatrixDim d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= d.rows) return;

  //copy to output and find max...
  double max = -1e20;
  double sum = 0.0;
  for(int i=0; i<d.cols; i++) {
    if(max < x[i+j*d.stride]) max = x[i+j*d.stride];
    y[i+j*d.stride] = x[i+j*d.stride];
  }
  //subtract max, apply exp, sum up...
  for(int i=0; i<d.cols; i++) {
    y[i+j*d.stride] = exp(y[i+j*d.stride] - max);
    sum += y[i+j*d.stride];
  }
  //normalize by sum...
  for(int i=0; i<d.cols; i++) {
    y[i+j*d.stride] /= sum;
  }
}




template<typename T>
__device__
static T _max_reduce(T buffer[]) {

  // Total number of active threads
  int nTotalThreads = blockDim.x;	
  __syncthreads();

  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      T temp = -1e20;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = buffer[threadIdx.x + halfPoint];
      }
      if (temp > buffer[threadIdx.x]) buffer[threadIdx.x] = temp;
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return buffer[0];
}




template<typename T>
__device__
static T _sum_reduce(T buffer[]) {

  // Total number of active threads
  int nTotalThreads = blockDim.x;	
  __syncthreads();

  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      T temp = 0.0;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = buffer[threadIdx.x + halfPoint];
      }
      buffer[threadIdx.x] += temp;
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return buffer[0];
}



template<typename T>
__global__
static void _softmax_reduce(T*y, const T*x, MatrixDim d) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(blockIdx.x > 0) return;
  if(blockDim.y > 1) return;

  __shared__ T row_data[256];
  __shared__ T aux[256];

  //copy the input to row_data
  row_data[i] = x[i+j*d.stride];
  __syncthreads();

  //copy input to aux
  aux[i] = row_data[i];
  __syncthreads();
  //get the maximum value
  T max = _max_reduce(aux);
  __syncthreads();

  //calculate exp(data-max)
  row_data[i] = exp(row_data[i]-max);
 
  //copy the values to aux
  aux[i] = row_data[i];
  __syncthreads();
  //get the sum
  T sum = _sum_reduce(aux);
  __syncthreads();

  //divide the values
  row_data[i] /= sum;
  //copy out
  y[i+j*d.stride] = row_data[i];

}



template<typename T>
__global__
static void _expand(T* y, const T* x, const int* off, MatrixDim d_out, MatrixDim d_in)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d_out.stride;
  if( i < d_out.cols  && j < d_out.rows ) {
    int src_col = i % d_in.cols;
    int src_row = j + off[i / d_in.cols];
    if(src_row < 0) src_row = 0;
    if(src_row >= d_in.rows) src_row = d_in.rows-1;
    y[index] = x[src_col + src_row*d_in.stride];
  }
}


template<typename T>
__global__
static void _rearrange(T* y, const T* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d_out.stride;
  if( i < d_out.cols  && j < d_out.rows ) {
    int src_col = copy_from[i];
    if(src_col >= 0 && src_col < d_in.cols) {
      y[index] = x[src_col + j*d_in.stride];
    } else {
      y[index] = 1.0/0.0;
    }
  }
}


template<typename T>
__global__
static void _randomize(T* y, const T* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d_out.stride;
  if( i < d_out.cols  && j < d_out.rows ) {
    int src_row = copy_from[j];
    y[index] = x[i + src_row*d_in.stride];
  }
}


template<typename T>
__global__
static void _check_class(const T* out, const T* des, float* match, MatrixDim d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(j>0) return;

  if(i<d.rows) {
    int out_id = -1, des_id = -2;
    T out_max = -1e20, des_max = -1e20;

    for(int k=0; k<d.cols; k++) {
      T val = out[k + i*d.stride];
      if(val > out_max) { out_max = val; out_id = k; }
    }
    for(int k=0; k<d.cols; k++) {
      T val = des[k + i*d.stride];
      if(val > des_max) { des_max = val; des_id = k; }
    }
    
    match[i] = ((out_id == des_id)?1:0);
  }
}


template<typename T>
__device__
static int _max_id_reduce(T val[],int idx[]) {

  // Total number of active threads
  int nTotalThreads = blockDim.x;	
  __syncthreads();

  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      T temp = -1e20;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = val[idx[threadIdx.x + halfPoint]];
      }
      if (temp > val[idx[threadIdx.x]]) idx[threadIdx.x]=idx[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return idx[0];
}






template<typename T>
__global__
static void _check_class_reduce(const T* out, const T* des, float* match, MatrixDim d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(blockIdx.x > 0) return;
  if(blockDim.y != 1) return;

  __shared__ T value[256];
  __shared__ int index[256];

  value[threadIdx.x] = out[i+j*d.stride];
  index[threadIdx.x] = threadIdx.x;
  __syncthreads();

  int out_max = _max_id_reduce(value,index);
  __syncthreads();

  value[threadIdx.x] = des[i+j*d.stride];
  index[threadIdx.x] = threadIdx.x;
  __syncthreads();
  
  int des_max = _max_id_reduce(value,index);
  __syncthreads();

  if(threadIdx.x == 0) {
    match[j] = ((out_max == des_max)?1:0);
  }
}




/*
 * ANSI-C wrappers of CUDA kernels
 */

/*
 * float 
 */

/*
 * CuMatrix
 */
void cudaF_set_const(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d); 
}

void cudaF_apply_log(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d); 
}

void cudaF_apply_mask(dim3 Gr, dim3 Bl, float* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask); 
}

void cudaF_scale_cols(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _scale_cols<<<Gr,Bl>>>(mat,scale,d); 
}

void cudaF_scale_rows(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _scale_rows<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_add_scaled(dim3 Gr, dim3 Bl, float alpha, const float* A, float beta, float* dst, MatrixDim d) {
  _add_scaled<<<Gr,Bl>>>(alpha,A,beta,dst,d); 
}

void cudaF_add_scaled_row(dim3 Gr, dim3 Bl, float alpha, const float* row, float beta, float* dst, MatrixDim d) {
  _add_scaled_row<<<Gr,Bl>>>(alpha,row,beta,dst,d); 
}

void cudaF_mul_elem(dim3 Gr, dim3 Bl, float*mat, const float*A, MatrixDim d) {
  _mul_elem<<<Gr,Bl>>>(mat,A,d); 
}

/*
 * CuVector
 */
void cudaF_add_col_sum(size_t Gr, size_t Bl, float alpha, const float* mat, float beta, float* vec, MatrixDim d) {
  _add_col_sum<<<Gr,Bl>>>(alpha,mat,beta,vec,d); 
}

void cudaF_add_col_sum_reduce(dim3 Gr, dim3 Bl, float alpha, const float* mat, float beta, float* vec, MatrixDim d) {
  _add_col_sum_reduce<<<Gr,Bl>>>(alpha,mat,beta,vec,d); 
}

/*
 * cu::
 */
void cudaF_sigmoid (dim3 Gr, dim3 Bl, float *y, const float*x, MatrixDim d) {
  _sigmoid<<<Gr,Bl>>>(y, x, d); 
}

void cudaF_diff_sigmoid (dim3 Gr, dim3 Bl, float*eout, const float*e, const float*y, MatrixDim d) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d);
}

void cudaF_softmax (size_t Gr, size_t Bl, float*y, const float*x, MatrixDim d) { 
  _softmax<<<Gr,Bl>>>(y, x, d); 
}

void cudaF_softmax_reduce (dim3 Gr, dim3 Bl, float*y, const float*x, MatrixDim d) { 
  _softmax_reduce<<<Gr,Bl>>>(y, x, d); 
}


void cudaF_expand(dim3 Gr, dim3 Bl, float* y, const float* x, const int* off, MatrixDim d_out, MatrixDim d_in) {
  _expand<<<Gr,Bl>>>(y,x,off,d_out,d_in); 
}


void cudaF_rearrange(dim3 Gr, dim3 Bl, float* y, const float* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _rearrange<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}

  
void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in) { 
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}


void cudaF_check_class(size_t Gr, size_t Bl, const float* out, const float* des, float* match, MatrixDim d) { 
  _check_class<<<Gr,Bl>>>(out,des,match,d); 
}

void cudaF_check_class_reduce(dim3 Gr, dim3 Bl, const float* out, const float* des, float* match, MatrixDim d) { 
  _check_class_reduce<<<Gr,Bl>>>(out,des,match,d); 
}

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float* wei, float* grad, float l1, float lr, MatrixDim d) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d); 
}

