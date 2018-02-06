// Copyright      2018  Zhehuai Chen

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "cuda-decoder-utils.h"
#include "decoder/cuda-decoder.h"

namespace kaldi {

#define CudaVector  CudaDecoder::CudaVector
#define CudaMergeVector CudaDecoder::CudaMergeVector
  typedef CudaDecoder::Token Token;
  typedef CudaDecoder::StateId StateId;
  typedef CudaDecoder::TokenState TokenState;
  typedef CudaDecoder::CostType CostType;
  typedef CudaDecoder::TokenLookupElem TokenLookupElem;
  typedef CudaDecoder::TokenVector TokenVector;
  typedef CudaDecoder::TokenMergeVector TokenMergeVector;
  typedef CudaDecoder::processTokens_params processTokens_params;


// for speedup purpose, make them inline (5% 0.165->0.158)
inline DEVICE uint64_t pack (float cost, int ptr) {
  //assert (!isnan(cost));
  //assert (ptr >= 0 && ptr < 1L<<32);
  uint32_t i_cost = *(uint32_t *)&cost;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0xFFFFFFFF;
  else
    i_cost = i_cost ^ 0x80000000;
  return (uint64_t)i_cost << 32 | ptr;
}

// Unpacks a probability.
inline DEVICE float unpack_cost (uint64_t packed) {
  uint32_t i_cost = packed >> 32;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0x80000000;
  else
    i_cost = i_cost ^ 0xFFFFFFFF;
  return *(float *)&i_cost;
}

// Unpacks a back-pointer.
inline DEVICE int unpack_ptr (uint64_t packed) {
  //assert (!(packed & 0x80000000));
  return packed & 0x7FFFFFFF;
}


inline  DEVICE void load16(void *a, const void *b) {
    const ulong2 *src = reinterpret_cast<const ulong2*>(b);
    ulong2 &dst = *reinterpret_cast<ulong2*>(a);
    asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
  }
  
inline  DEVICE void store16(void *a, const void *b) {
    const ulong2 src = *reinterpret_cast<const ulong2*>(b);
    asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
  }

  
inline  DEVICE void store32(void *a, const void *b) {
    //const ulong4 src = *reinterpret_cast<const ulong4*>(b);
    //asm("st.global.v4.u64 [%0], {%1,%2,%3,%4};" :: "l"(a), "l"(src.x), "l"(src.y),
    //  "l"(src.z), "l"(src.w));
    memcpy(a, b, 32);
  }

inline DEVICE void atomicMin(double *address, double val) {
  unsigned long long *address_ull = (unsigned long long *)address;

  double minval = *address;

  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val))); //write minimum and read back value
  } //if the new value is < the minimum I wrote I need to try again.
}
inline DEVICE void atomicMin(float *address, float val) {
  unsigned int *address_ui = (unsigned int  *)address;

  float minval = *address;

  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val))); //write minimum and read back value
  } //if the new value is < the minimum I wrote I need to try again.
}

// end of "for speedup purpose, make them inline (5% 0.165->0.158)"


//private, as we need to instantiate them  
template<typename T> 
  inline DEVICE void swap(T &a, T &b) {
    T c = a;
    a = b;
    b = c;
  }

/******************************************CudaVector Implementation*******************************/
template<typename T>
  HOST DEVICE inline T& CudaVector<T>::operator[](uint32_t idx) { 
#ifdef __CUDA_ARCH__
    assert(idx<*count_d);
    return mem_d[idx];
#else
    assert(idx<*count_h);
    return mem_h[idx];
#endif
  }

template<typename T>
  HOST DEVICE inline const T& CudaVector<T>::operator[](uint32_t idx) const { 
#ifdef __CUDA_ARCH__
    assert(idx<*count_d);
    return mem_d[idx];
#else
    assert(idx<*count_h);
    return mem_h[idx];
#endif
  } 

template<typename T>
  inline void CudaVector<T>::allocate(uint32_t max_size, 
     uint32_t* icount_h, uint32_t* icount_d, T* mem_d, T* mem_h) {
    this->max_size=max_size;
    alloc_size=0;

    if (icount_h) this->count_h=icount_h;
    else {
      cudaMallocHost(&this->count_h,sizeof(uint32_t));
    }
      if (icount_d) this->count_d=icount_d;
      else {
        alloc_size+=sizeof(uint32_t);
        cudaMalloc(&this->count_d, sizeof(uint32_t));
      }
      cudaMemset(this->count_d, 0,sizeof(uint32_t));
      *count_h=0;

      if (mem_d) {
        this->mem_d=mem_d;        
      } else {
        alloc_size+=max_size*sizeof(T);
        cudaMalloc(&this->mem_d,max_size*sizeof(T));
      }
      if (mem_h) {
        this->mem_h=mem_h;        
      } else {
        cudaMallocHost(&this->mem_h,max_size*sizeof(T));
      }
    }

  template<typename T>
    inline size_t CudaVector<T>::getCudaMallocBytes() {
      return alloc_size;
    }

  template<typename T>
    inline void CudaVector<T>::free(bool create_outside) { 
      cudaFreeHost(mem_h);
      if (!create_outside) {
        cudaFree(mem_d); 
      }
      cudaFreeHost(count_h);
      cudaFree(count_d);       
    }


  template<typename T>
    inline void CudaVector<T>::copy_all_to_host(cudaStream_t stream) {
      cudaStreamSynchronize(stream);
      cudaMemcpy(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(mem_h,mem_d,*count_h*sizeof(T),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_all_to_device(cudaStream_t stream) {
      cudaStreamSynchronize(stream);
      cudaMemcpyAsync(count_d,count_h,sizeof(int32),cudaMemcpyHostToDevice);
      cudaMemcpyAsync(mem_d,mem_h,*count_h*sizeof(T),cudaMemcpyHostToDevice, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_size_to_host(cudaStream_t stream) {
      cudaMemcpyAsync(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_size_to_device(cudaStream_t stream) {
      cudaMemcpyAsync(count_d,count_h,sizeof(int32),cudaMemcpyHostToDevice, stream);
    }
  
template<typename T>
    inline void CudaVector<T>::copy_data_to_host(cudaStream_t stream, T* to_buf, bool copy_size) {
      if (!to_buf) {
        to_buf=mem_h;
      }
      if (copy_size) cudaMemcpy(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(to_buf,mem_d,*count_h*sizeof(T),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_data_to_device(cudaStream_t stream) {
      cudaMemcpyAsync(mem_d,mem_h,*count_h*sizeof(T),cudaMemcpyHostToDevice, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_data_to_device(int size, T* mem_in_d, cudaStream_t stream) {
      cudaMemcpyAsync(mem_d+*count_d*sizeof(T),mem_in_d,size*sizeof(T),cudaMemcpyDeviceToDevice, stream);
      *count_d+=size;
    }



  //Note:  This will cause page faults back and forth when we switch from host to device.
  template<typename T>
    HOST DEVICE inline uint32_t CudaVector<T>::size() const 
    {
#ifdef __CUDA_ARCH__
      return *count_d; 
#else
      return *count_h;
#endif
    }

  template<typename T> 
    HOST DEVICE inline uint32_t CudaVector<T>::push_back(const T &val) { 
#ifdef __CUDA_ARCH__
      assert(*count_d<max_size);
      uint32_t idx = atomicAdd(count_d,1);
      mem_d[idx]=val; 
#else
      assert(*count_h<max_size);
      uint32_t idx = (*count_h)++;
      mem_h[idx]=val; 
#endif
      return idx;
    }
  template<typename T> 
    HOST DEVICE inline void CudaVector<T>::clear(cudaStream_t stream) { 
#ifdef __CUDA_ARCH__
      *count_d = 0;
#else
      *count_h = 0; 
      cudaMemsetAsync(count_d,0,sizeof(int32),stream); 
#endif
    }
  template<typename T> 
    HOST DEVICE inline int CudaVector<T>::get_idx_from_addr(T* addr) { 
#ifdef __CUDA_ARCH__
      int ret=addr-mem_d;
      assert(ret<*count_d&&ret>=0);
      return ret;
#else
      int ret=addr-mem_h;
      assert(ret<*count_h&&ret>=0);
      return ret;
#endif
    }
  template<typename T> 
    inline bool CudaVector<T>::empty() const { return size()==0; }
  template<typename T> 
    inline void CudaVector<T>::swap(CudaVector<T> &v) {
      std::swap(mem_h,v.mem_h);
      std::swap(mem_d,v.mem_d);
      std::swap(count_h,v.count_h);
      std::swap(count_d,v.count_d);
      std::swap(max_size,v.max_size);
    }
  /**************************************End CudaVector Implementation**********************************/

//end of "private, as we need to instantiate them  "

  template<typename T> 
    inline void CudaMergeVector<T>::swap(CudaMergeVector<T> &v) {
      CudaVector<T>::swap(v);
      std::swap(mem_buf_count_d,v.mem_buf_count_d);
      std::swap(mem_update_d,v.mem_update_d);
    }


template<typename T> 
  DEVICE inline int CudaMergeVector<T>::update(int i) {
    if (i>=*count_d) return 0;
    return mem_update_d[i];
  }
template<> 
DEVICE inline void CudaMergeVector<TokenState>::merge(void* token_per_arc, int* token_per_arc_update, int num_arcs, bool clear) {
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  int idx=tid;
  int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
  int batch=blockDim.x*gridDim.x; 
  if (rank0) {
    int acc=0;
    int i=0;
    mem_buf_acc_count_d[i]=acc;
    acc+=(mem_buf_count_d[i]);
    if (clear) mem_buf_count_d[i]=0;
    assert(acc<=max_size);
    *count_d=acc;
    mem_buf_acc_count_d[1]=acc;
  }
  __grid_sync_nv_internal(barrier_);
  int sz = mem_buf_acc_count_d[1]-mem_buf_acc_count_d[0];
  for(; idx < sz; idx += batch) {
    uint64_t* pack_v=mem_pack_buf_d[idx];
    int ptr=unpack_ptr(*pack_v);
    //assert(ptr<num_arcs);
    mem_update_d[(idx+mem_buf_acc_count_d[0])]=token_per_arc_update[ptr];
  #if 1
    if (token_per_arc_update[ptr]) token_per_arc_update[ptr]=0;
    else continue;
  #endif
    TokenState* to_ts=mem_d+(idx+mem_buf_acc_count_d[0]);
    Token* cur_tok=((Token *)token_per_arc)+ptr;
    Token* to_tok=to_ts->token;
    store16(to_tok, cur_tok);
    //memcpy(to_tok,cur_tok,sizeof(T));
  }    
}

template<typename T> 
DEVICE inline void CudaMergeVector<T>::clear_sub() {
  int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
  if (rank0) {
    memset(mem_buf_count_d, 0, sizeof(int)*(2));
  }
}


template<typename T> 
DEVICE inline void CudaMergeVector<T>::merge(void* undefined, int* token_per_arc_update, int num_arcs, bool clear) {
  assert(0);
}

  template<typename T> 
    DEVICE inline uint32_t CudaMergeVector<T>::push_back(const T &val, 
                                    uint64 *val_pack) { 
      uint32_t idx = atomicAdd(mem_buf_count_d,1);
      mem_d[idx]=val;
      mem_pack_buf_d[idx]=val_pack; 
      //CudaVector<T>::push_back(val); //do this is only for speedup in PNE; dont need to
      return idx;
    }

  template<typename T>
    inline void CudaMergeVector<T>::allocate(uint32_t max_size) {
      CudaVector<T>::allocate(max_size);

      cudaMalloc(&mem_pack_buf_d,sizeof(uint64_t*)*max_size);
      cudaMalloc(&mem_buf_d,sizeof(T)*max_size);
      cudaMalloc(&mem_update_d,sizeof(int)*max_size);
      cudaMemset(mem_update_d,0,sizeof(int)*max_size);
      cudaMalloc(&mem_buf_count_d,sizeof(int)*(2));
      cudaMalloc(&mem_buf_acc_count_d,sizeof(int)*(2));
      cudaMalloc(&barrier_,sizeof(int)*1);
    }

  template<typename T>
    inline size_t CudaMergeVector<T>::getCudaMallocBytes() {
      return CudaVector<T>::getCudaMallocBytes()+
        sizeof(uint32_t)*(1+2*(2))+max_size*(sizeof(T)+sizeof(uint64_t*)+sizeof(int));
    }

  template<typename T>
    inline void CudaMergeVector<T>::free() { 
      CudaVector<T>::free();
      cudaFree(mem_pack_buf_d);
      cudaFree(mem_buf_d);
      cudaFree(mem_update_d);
      cudaFree(mem_buf_count_d);
      cudaFree(mem_buf_acc_count_d);
      cudaFree(barrier_);
    }

  DEVICE inline void allocateAllTokens_function(CudaDecoder::TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaDecoder::TokenAllocator allocator) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<numStates; i+=blockDim.x*gridDim.x) {
      CudaDecoder::Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->prev_ = NULL;
      CudaDecoder::TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      elem.token_pack=pack(-FLT_MAX, 0);
      //store16(&current_tokens_lookup[i], &elem);
      memcpy(&current_tokens_lookup[i], &elem, sizeof(CudaDecoder::TokenLookupElem));
    }
  }
  __global__ void allocateAllTokens(CudaDecoder::TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaDecoder::TokenAllocator allocator, int *barrier) {
    allocateAllTokens_function(current_tokens_lookup,numStates,allocator);
     __grid_sync_nv_internal(barrier);
     if(blockIdx.x==0 && threadIdx.x==0) {
      allocator.advanceFront(numStates);
     }
  }

  DEVICE inline void allocateNewTokens_function(CudaDecoder::TokenLookupElem *current_tokens_lookup, CudaDecoder::TokenMergeVector cur_toks, CudaDecoder::TokenAllocator allocator) {
    int32 size = cur_toks.size();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x) {
      Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->prev_ = NULL;
      StateId state=cur_toks[i].state;
      TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      elem.token_pack=pack(-FLT_MAX, 0);
      memcpy(&current_tokens_lookup[state], &elem, sizeof(CudaDecoder::TokenLookupElem));
      //store16(&current_tokens_lookup[state], &elem);
    }
  }

  
  void CudaDecoder::TokenAllocator::prefetch_next_to_device(cudaStream_t stream) {
    prefetch_next_to_device(stream,prefetch_size);
  }

  void CudaDecoder::TokenAllocator::prefetch_next_to_device(cudaStream_t stream, int count) {
    int front = *front_h;
    //clamp to maximum size
    if(count>size-front)
      count = size-front;

#ifdef MEMADVISE
    cudaMemPrefetchAsync(tokens_allocation+front,sizeof(Token)*count,device,stream);  
#endif
  }

  void CudaDecoder::TokenAllocator::prefetch_allocated_to_host(cudaStream_t stream) {
#ifdef MEMADVISE
    cudaMemPrefetchAsync(tokens_allocation,sizeof(Token)* *front_h,cudaCpuDeviceId,stream);  
#endif
  }

  size_t CudaDecoder::TokenAllocator::getCudaMallocManagedBytes() {
    return bytes_cudaMallocManaged;
  }

  void CudaDecoder::TokenAllocator::reset() {
    *front_h=0;
    cudaMemset(front_d,0,sizeof(int));
  }

  void CudaDecoder::TokenAllocator::initialize(uint32_t size)  {
    cudaGetDevice(&device);
    prefetch_size=250000;

    this->size = size;

    //managed so getBestPath can easily access this data in the end
    cudaMallocManaged((void**)&tokens_allocation,sizeof(Token)*size);  
    bytes_cudaMallocManaged=sizeof(Token)*size;

    cudaMalloc((void**)&front_d,sizeof(uint32_t)); 
    cudaMallocHost((void**)&front_h,sizeof(uint32_t)); 

#ifdef MEMADVISE
    //If we do this we get faster perf as long as we don't over subscribe
    cudaMemAdvise(tokens_allocation,sizeof(Token)*size,cudaMemAdviseSetPreferredLocation,device);
    cudaMemPrefetchAsync(tokens_allocation,sizeof(Token)*size,device);  //force pages to allocate now
#endif

    reset();
  }

  void CudaDecoder::TokenAllocator::finalize() {
    printf("TokenAllocator::finalize()\n");
    cudaFree(tokens_allocation);
    cudaFree(front_d);
    cudaFreeHost(front_h);
  }

  DEVICE inline CudaDecoder::Token* CudaDecoder::TokenAllocator::getToken(uint32_t offset) {
    int idx = *front_d + offset;
    return &tokens_allocation[idx];
  }

  DEVICE inline void CudaDecoder::TokenAllocator::advanceFront(uint32_t num) {
    int front = *front_d + num;
    //assert(front<size);
    
    *front_d=front;
    *front_h=front;
  }


  CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config): fst_(fst), beam_(config.beam), bytes_cudaMalloc(0), bytes_cudaMallocManaged(0) {
    printf("CudaDecoder Constructor\n");
    int device;
    cudaGetDevice(&device);
    cudaCheckError();

    if (verbose>4) cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1e7);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);

    total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount * config.gpu_fraction;

    allocator.initialize(config.max_tokens);

    bytes_cudaMallocManaged+=allocator.getCudaMallocManagedBytes();
    cur_toks_.allocate(config.max_tokens_per_frame);
    prev_toks_.allocate(config.max_tokens_per_frame);
    bytes_cudaMalloc+=cur_toks_.getCudaMallocBytes()+prev_toks_.getCudaMallocBytes();

    cudaEventCreateWithFlags(&event_pt,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_pt_old,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_ll,cudaEventDisableTiming);
    cudaCheckError();

    cudaStreamCreateWithFlags(&stream_comp, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_copy, cudaStreamNonBlocking);
    cudaStreamCreateWithPriority(&stream_ll, cudaStreamNonBlocking, -1);

    cudaMalloc(&pe_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&cidx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&cidx2_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&ne_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&ne_queue_d, sizeof(int)*config.max_tokens_per_frame); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&l_ne_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&fb_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&barrier_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);

    cudaMemset(pe_idx_d,0,sizeof(int));
    cudaMemset(ne_idx_d,0,sizeof(int));
    cudaMemset(cidx_d,0,sizeof(int));
    cudaMemset(cidx2_d,0,sizeof(int));
    cudaMemset(l_ne_idx_d,0,sizeof(int));
    cudaMemset(fb_idx_d,0,sizeof(int));
    cudaMemset(barrier_d,0,sizeof(int));

    cudaMalloc(&cutoff_d, sizeof(CostType)); bytes_cudaMalloc+=sizeof(CostType);
    cudaMalloc(&modified_d, sizeof(int)*2); bytes_cudaMalloc+=sizeof(CostType)*2;

    cudaMalloc(&token_locks_d,sizeof(int)*fst_.numStates);  bytes_cudaMalloc+=sizeof(int)*fst_.numStates;
    cudaMemset((void*)token_locks_d,0,sizeof(int)*fst_.numStates);

    cudaMalloc((void**)&current_tokens_lookup_d,sizeof(TokenLookupElem)*fst_.numStates); bytes_cudaMalloc+=sizeof(TokenLookupElem)*fst_.numStates;

    cudaMallocHost(&loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));  
    cudaMallocHost(&loglikelihoods_old_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));

    cudaMalloc((void**)&loglikelihoods_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);
    cudaMalloc((void**)&loglikelihoods_old_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);

    cudaMalloc((void**)&clock_buf_d,sizeof(uint64)*100); bytes_cudaMalloc+=sizeof(uint64)*(100);
    cudaMemset(clock_buf_d,0,sizeof(uint64)*100);

    cudaMalloc((void**)&token_per_arc_d,sizeof(Token)*fst.NumArcs()); //temp solution
    cudaMalloc((void**)&token_per_arc_update_d,sizeof(int)*fst.NumArcs()); //temp solution
    cudaMemset(token_per_arc_update_d,0,sizeof(int)*fst.NumArcs()); //temp solution
    bytes_cudaMalloc+=sizeof(Token)*(fst.NumArcs());
    
    cudaStreamSynchronize(stream_comp);
    cudaStreamSynchronize(stream_copy);
    cudaStreamSynchronize(cudaStreamPerThread);

    //sgemm requires shared memory and we don't want cache config changing.  So set a device wide cache config.
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

    verbose=config.verbose;
    cudaCheckError();
  }

  CudaDecoder::~CudaDecoder() {

    printf("CUDA DECODER DESTRUCTOR\n");

    if (verbose>1) {
      uint64 t[20];
      cudaMemcpy(t,clock_buf_d,sizeof(t),cudaMemcpyDeviceToHost);
      cudaDeviceProp prop;
      int device;
      cudaGetDevice(&device);
      cudaGetDeviceProperties(&prop, device);
      for (int i=0;i<sizeof(t)/sizeof(uint64);i++) std::cout<<1.0/prop.clockRate*t[i]<<" ";
      std::cout<<"\n";
    }

    cur_toks_.free();
    prev_toks_.free();
    allocator.finalize();

    cudaFreeHost(loglikelihoods_h);
    cudaFreeHost(loglikelihoods_old_h);
    cudaFree(loglikelihoods_d);
    cudaFree(loglikelihoods_old_d);
    cudaFree(current_tokens_lookup_d);

    cudaFree(pe_idx_d);
    cudaFree(cidx_d);
    cudaFree(cidx2_d);
    cudaFree(ne_idx_d);
    cudaFree(ne_queue_d);
    cudaFree(l_ne_idx_d);
    cudaFree(fb_idx_d);
    cudaFree(barrier_d);

    cudaFree((void*)token_locks_d);
    cudaFree(cutoff_d);
    cudaFree(modified_d);

    cudaFree(clock_buf_d);
    cudaFree(token_per_arc_d);
    cudaFree(token_per_arc_update_d);


    cudaEventDestroy(event_pt);
    cudaEventDestroy(event_pt_old);
    cudaEventDestroy(event_ll);

    cudaStreamDestroy(stream_comp);
    cudaStreamDestroy(stream_copy);
    cudaStreamDestroy(stream_ll);

  }

  void CudaDecoder::PreProcessTokens() {
    PUSH_RANGE("PreProcessTokens",0)
    //before reset, we should update tid2arc_d for the next frame
    num_frames_decoded_++;
      //cudaStreamSynchronize(stream_comp);
      //assert(cur_toks_.size());
      ////if (verbose>4) {
      ////  int * tmp;
      ////cudaMallocHost((void**)&tmp,10*sizeof(int)); 
      ////cudaMemcpy(tmp,tok2scansum_numarc_d,sizeof(int)*10,cudaMemcpyDeviceToHost);
      ////cudaCheckError();
      ////KALDI_LOG<<tmp[0]<<" "<<tmp[1]<< " "<<tmp[2];
      ////cudaFree(tmp);
      ////}
      //cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
      //  tok2scansum_numarc_d, tok2scansum_numarc_d, cur_toks_.size()+1, stream_comp);
      //cudaCheckError();
      //if (verbose>4) {
      //  int * tmp;
      //cudaMallocHost((void**)&tmp,10*sizeof(int)); 
      //cudaMemcpy(tmp,tok2scansum_numarc_d,sizeof(int)*10,cudaMemcpyDeviceToHost);
      //KALDI_LOG<<tmp[0]<<" "<<tmp[1]<< " "<<tmp[2];
      //cudaFree(tmp);
      //cudaCheckError();
      //}

#ifndef MEMADVISE
      //no need to prefetch if we have done a memadvise
      allocator.prefetch_next_to_device(cudaStreamPerThread);
#endif

      //TODO prefetch here

    cur_toks_.swap(prev_toks_);

    POP_RANGE
  }

  bool CudaDecoder::Decode(DecodableInterface *decodable) {
    nvtxRangePushA("CudaDecoder::Decode");

    InitDecoding();


    ComputeLogLikelihoods(decodable);

    while( !decodable->IsLastFrame(num_frames_decoded_ - 1)) {

      if (verbose>4) KALDI_LOG << num_frames_decoded_<<std::endl;

      PreProcessTokens();

      ProcessTokens();

      //computes log likelihoods for the next frame
      ComputeLogLikelihoods(decodable);
    }

    cur_toks_.copy_all_to_host(stream_comp);
    cudaStreamSynchronize(stream_comp);

    nvtxRangePop();

    return (!cur_toks_.empty());
  }

  DEVICE inline Token* FindOrAddTokenArc(processTokens_params& params,
    StateId nextstate, CostType total_cost, CostType acoustic_cost,
    TokenState* ts, bool use_sub, uint64_t **token_pack, int* update) {
    //TokenLookupElem lookup_elem;
    //load16(&lookup_elem, &params.current_tokens_lookup[nextstate]);
    TokenLookupElem& lookup_elem = params.current_tokens_lookup[nextstate];
    Token *cur_tok = lookup_elem.token;  
    //check if token is active or not.  Double check the lock.
    if(lookup_elem.active==0 && atomicCAS(&lookup_elem.active,0,1)==0) {        //grab sentinal to see who gets to add to cur_toks list
      //if havent seen, add into hash
      *update=1;
      if (use_sub) 
        params.cur_toks.push_back(TokenState(cur_tok,nextstate), 
        &lookup_elem.token_pack);
      else params.cur_toks.push_back(TokenState(cur_tok,nextstate));
    }
      if (use_sub) *token_pack=&lookup_elem.token_pack;

    return cur_tok;  
  }

  __global__ void addOneToken(processTokens_params params, CudaDecoder::StateId state) {
    uint64_t* token_pack;
    int j=0;
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    Token* cur_tok=FindOrAddTokenArc(params, state, 0, //add first token
      0, NULL,true, &token_pack, params.token_per_arc_update+j);
    uint64_t new_token_pack=pack(0, j);
    Token* cur_te=params.token_per_arc+j;
    params.token_per_arc_update[j]=1;
    store16(cur_te, &(Token(0, NULL, j)));
    atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
    params.cur_toks.merge(params.token_per_arc,params.token_per_arc_update, params.numArcs, false);
  }

  //putting this into a kernel to avoid extra latency of a memory copy
  __global__ void initializeCutoff(CudaDecoder::CostType *cutoff) {
    *cutoff = INFINITY;
  }

  void CudaDecoder::InitDecoding() {
    printf("CUDA DECODER InitDecoding\n");
    PUSH_RANGE("InitDecoding",1)
    cudaCheckError();
    // clean up from last time:
    ClearToks(cur_toks_);
    ClearToks(prev_toks_);
    cudaCheckError();
    
    allocator.reset();
    int threads=64;
    int blocks=DIV_ROUND_UP(total_threads,threads);
    
    //start moving these / allocating them on the device
    allocator.prefetch_next_to_device(stream_comp, fst_.numStates+5000);

    allocateAllTokens<<<blocks,threads,0,stream_comp>>>(current_tokens_lookup_d, fst_.numStates, allocator, barrier_d);

    // initialize decoding:
    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);

    cudaCheckError();
    processTokens_params params;
    initParams(params);
    addOneToken<<<1,1,0,stream_comp>>>(params, start_state);
    cudaCheckError();

    initializeCutoff<<<1,1,0,stream_comp>>>(cutoff_d);

    num_frames_decoded_ = 0;
    ProcessNonemitting();

    cur_toks_.copy_size_to_host(stream_comp); //for PreProcessTokens
    POP_RANGE
  }

  bool CudaDecoder::ReachedFinal() const {
    for (int i=0;i<cur_toks_.size();i++) {
      TokenState ts = cur_toks_[i];

      if (ts.token->cost_ != std::numeric_limits<BaseFloat>::infinity() &&
          fst_.Final(ts.state) != StdWeight::Zero())
        return true;
    }

    return false;
  }

  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
    nvtxRangePushA("GetBestPath");


    fst_out->DeleteStates();
    Token *best_tok = NULL;
    bool is_final = ReachedFinal();
    
    if (!is_final) {
      for(int i=0;i<cur_toks_.size();i++) {
        TokenState ts = cur_toks_[i];
        Token *tok = ts.token;
        if(best_tok==NULL || *best_tok < *tok) {
          best_tok = tok;
        }
      }
    } else {
      CostType infinity =std::numeric_limits<CostType>::infinity(),
               best_cost = infinity;
      for(int i=0;i<cur_toks_.size();i++) {
        TokenState ts = cur_toks_[i];
        Token  *tok = ts.token;
        StateId state = ts.state;
        CostType this_cost = tok->cost_ + fst_.Final(state);
        if (this_cost != infinity && this_cost < best_cost) {
          best_cost = this_cost;
          best_tok = tok;
        }
      }
    }

    if (best_tok == NULL) {
      nvtxRangePop();
      return false;  // No output.
    }

    int count=0;

    //for each token in reverse order
    //add arc to list
    std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
    for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
      count++;
      Token &t=*tok;

      uint32_t arc_idx=t.arc_index_;

      LatticeArc arc(fst_.arc_ilabels_h[arc_idx], fst_.arc_olabels_h[arc_idx], LatticeWeight(fst_.arc_weights_h[arc_idx], 0), fst_.arc_nextstates_h[arc_idx]);

      arcs_reverse.push_back(arc);
    }
    KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

    //for each arc in reverse
    //generate new fst
    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);
    for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
      LatticeArc arc = arcs_reverse[i];
      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
    if (is_final && use_final_probs)
      fst_out->SetFinal(cur_state,
          LatticeWeight(fst_.Final(fst_.arc_nextstates_h[best_tok->arc_index_]),
            0.0));
    else
      fst_out->SetFinal(cur_state, LatticeWeight::One());
    fst::RemoveEpsLocal(fst_out);
    nvtxRangePop();
    return true;
  }

  void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
    nvtxRangePushA("ComputeLogLikelihoods");

    int32 frame = num_frames_decoded_;

    std::swap(loglikelihoods_h,loglikelihoods_old_h); //double buffering so we don't overwrite loglikelihoods_h before it is copied down
    std::swap(loglikelihoods_d,loglikelihoods_old_d); //double buffer

    //We really only need about 10% of these but finding out which 10% is more expensive then just computing all of them
    //Computing them inline in the next loop leads to lots of redundant computation
    decodable->ComputeLogLikelihoods(loglikelihoods_h,frame,fst_.max_ilabel+1);

    //copying in another stream to overlap transfer with compute
    cudaMemcpyAsync(loglikelihoods_d,loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1),cudaMemcpyHostToDevice, stream_ll);

    cudaEventRecord(event_ll,stream_ll);  //mark log likelihoods are copied down to the device
    cudaStreamWaitEvent(stream_comp,event_ll,0); //ensure logliklihoods_d is updated before consuming

    nvtxRangePop();
  }

  void CudaDecoder::initParams(processTokens_params& params) {
    params.prev_toks=prev_toks_;
    params.cur_toks=cur_toks_;
    params.allocator=allocator;
    params.e_offsets=fst_.e_offsets_d;
    params.ne_offsets=fst_.ne_offsets_d;
    params.arc_ilabels=fst_.arc_ilabels_d;
    params.arc_weights=fst_.arc_weights_d;
    params.arc_nextstates=fst_.arc_nextstates_d;
    params.cutoff=cutoff_d;
    params.loglikelihoods=loglikelihoods_d;
    params.current_tokens_lookup=current_tokens_lookup_d;
    params.token_locks=token_locks_d;
    params.modified=modified_d;
    params.beam=beam_;
    params.pe_idx=pe_idx_d;
    params.ne_idx=ne_idx_d;
    params.ne_queue=ne_queue_d;
    params.l_ne_idx=l_ne_idx_d;
    params.fb_idx=fb_idx_d;
    params.barrier=barrier_d;
    params.verbose=verbose;
    params.frame=num_frames_decoded_;

    params.tid2tok=tid2tok_d;
    params.tok2scansum_numarc=tok2scansum_numarc_d;
    params.clock_buf=clock_buf_d;
    params.tid2arc=tid2arc_d;
    params.max_arcs_per_frame_search=max_arcs_per_frame_search_;    
    params.token_per_arc=token_per_arc_d;
    params.token_per_arc_update=token_per_arc_update_d;
    params.numArcs=fst_.NumArcs();
    params.cidx=cidx_d;
    params.cidx2=cidx2_d;
  }


  void CudaDecoder::ClearToks(TokenMergeVector &toks) {
    //cannot acctually delete tokens as they may still be connected to active tokens
    toks.clear(stream_comp);
  }

  

  //blockDim.x threads per token
  template<int blockDimx, int blockDimy>
  inline DEVICE void findBestCutoff_function(processTokens_params params) {
    typedef CudaDecoder::TokenState TokenState;
    typedef CudaDecoder::Token Token; 
    typedef CudaDecoder::StateId StateId;
    typedef CudaDecoder::CostType CostType;

    int threadIdxy = threadIdx.x / blockDimx;

    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CudaDecoder::CostType local_cutoff = INFINITY;
    int32 size = params.prev_toks.size(); 

    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) { 
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.fb_idx,1);      //get token index
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;  //Work complete
      
      TokenState ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.e_offsets[state], finish=params.e_offsets[state+1];
      
      int32 ilabel, ilabel_next;

      int j=start+group.thread_rank();

      if(j<finish) {
        ilabel_next = params.arc_ilabels[j];
      }
      int nextj;

      for(j;j<finish;j=nextj) {
        nextj = j+blockDimx;
        ilabel = ilabel_next;
        if(nextj<finish) {
          ilabel_next = params.arc_ilabels[nextj];
        }
        
        BaseFloat acoustic_cost = -params.loglikelihoods[ilabel]; //TODO can I prefetch this?
        CostType weight = params.arc_weights[j];
        
        CudaDecoder::CostType total_cost = tok->cost_ + weight + acoustic_cost + params.beam;

        if(total_cost<local_cutoff)
          local_cutoff = total_cost;
      }
    }

    //TODO reduce inside block first?
    if(local_cutoff!=INFINITY) {
      atomicMin(params.cutoff, local_cutoff);
    }
  }

  //blockDim.x threads per token
  template<int blockDimx, int blockDimy>
  inline DEVICE void processEmittingTokens_function(processTokens_params params) {
    typedef CudaDecoder::TokenState TokenState;
    typedef CudaDecoder::Token Token; 
    typedef CudaDecoder::StateId StateId;
    typedef CudaDecoder::CostType CostType;
    typedef CudaDecoder::TokenLookupElem TokenLookupElem; 
    int threadIdxy = threadIdx.x / blockDimx;
    
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CostType cutoff=*params.cutoff;
    int32 size = params.prev_toks.size();
    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.pe_idx,1);      //get token index
        if (params.verbose>3 && i%1000==0) {
          printf("E: %i %i %i\n", i, threadIdx.x, blockIdx.x);
        }
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;

      TokenState ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.e_offsets[state], finish=params.e_offsets[state+1];
      int32 ilabel, ilabel_next;  //prefetch ilabel since it leads to a dependent load

      int j=start+group.thread_rank();

      if(j<finish) {
        ilabel_next = params.arc_ilabels[j];
      }
      int nextj;

      for(j;j<finish;j=nextj) {
        nextj = j+blockDimx;

        ilabel = ilabel_next;

        if(nextj<finish) {
          ilabel_next = params.arc_ilabels[nextj];
        }
        BaseFloat acoustic_cost = -params.loglikelihoods[ilabel];  //TODO can I prefetch this?  
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        CostType total_cost = tok->cost_ + weight + acoustic_cost;

        if(total_cost<=cutoff) 
        {
          uint64_t* token_pack;
          TokenState *next_ts=NULL;
          //get cur_tok&token_pack addr
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
          acoustic_cost, &ts, true, &token_pack, params.token_per_arc_update+j);
          //get cur_te&new_token_pack
          uint64_t new_token_pack=pack(-total_cost, j);
          uint64_t ret=atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
          if (ret<new_token_pack) {
            Token* cur_te=params.token_per_arc+j;
            store16(cur_te, &(Token(acoustic_cost+weight, tok, j)));
            params.token_per_arc_update[j]=1;
          }
        } //end total_cost<=cutoff
      } //end arc loop
    } //end token loop
    __grid_sync_nv_internal(params.barrier);
    params.cur_toks.merge(params.token_per_arc,params.token_per_arc_update, params.numArcs, false);
  }
    template<int blockDimx, int blockDimy>
  DEVICE __inline__ void processNonEmittingTokens_function(processTokens_params &params, CudaDecoder::CostType cutoff, uint32_t size,  volatile int *modified, bool aggregate=false) {

    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    int& cidx=*params.cidx;
    int& cidx2=*params.cidx2;
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    if (aggregate) {
      for (tid;tid<size;tid+=blockDim.x*gridDim.x) {
        if(params.cur_toks.update(tid)) {
          int i=atomicAdd(&cidx,1);      //get changed token index for faster NE proc
          if (i>=size) break;
          params.ne_queue[i]=tid;
        }
      }
      __grid_sync_nv_internal(params.barrier);
    }

    if (params.verbose>3&&threadIdx.x==0 && blockIdx.x==0) printf("PNE: %i %i %i\n",params.frame, params.cur_toks.size(), cidx);

    int threadIdxy = threadIdx.x / blockDimx;

    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i,j;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
      if (aggregate) {
          j=atomicAdd(&cidx2,1);      //get token index
          if (j>=cidx) i=size; // to exit
          else i=params.ne_queue[j];
      } else {
          i=atomicAdd(params.ne_idx,1);      //get token index
      }
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(aggregate&&i>=size) break;
      if(aggregate==0&&i>=params.cur_toks.size()) break;
      
      TokenState& ts = params.cur_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;
      
      uint32_t start=params.ne_offsets[state], finish=params.ne_offsets[state+1];
      for(int j=start+group.thread_rank();j<finish;j+=blockDimx) {
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        Token next_tok = Token(weight, tok, j);

        CostType total_cost = tok->cost_ + weight;

      if (params.verbose>4) printf("D: %i %i %i %i %i \n",threadIdx.x, threadIdx.y, j, blockIdx.x,i);
        if (next_tok.cost_ <= cutoff) {
          TokenState *next_ts=NULL;
          uint64_t* token_pack;
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
            0, &ts,  true, &token_pack, params.token_per_arc_update+j);
          uint64_t new_token_pack=pack(-total_cost, j);
          uint64_t ret=atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
          if (ret<new_token_pack) {
            Token* cur_te=params.token_per_arc+j;
            store16(cur_te, &(Token(weight, tok, j)));
            params.token_per_arc_update[j]=1;
            (*modified) = true;
          }
        }
      }
    }
    __grid_sync_nv_internal(params.barrier);
    params.cur_toks.merge(params.token_per_arc,params.token_per_arc_update, params.numArcs, false);
    if (threadIdx.x==0&&blockIdx.x==0) { cidx=cidx2=0; }
  }

  DEVICE inline uint64 gt(uint64 t1, uint64 t2) {
    if (t1>t2) return t1-t2;
    else return t2-t1;
  }
  __launch_bounds__(64,64)
  __global__ void processTokens_cg(processTokens_params params, bool is_init=false) {

    bool rank0 = blockIdx.x==0 && threadIdx.x==0;
    int p=0;
    int cnt_c=0;
    uint64 t[20];
    t[cnt_c]=clock64();

    if (!is_init) {
        findBestCutoff_function<32,2>(params);
        __grid_sync_nv_internal(params.barrier);
    }

    volatile int *modified0 = params.modified;    //modified flag for current iteration
    volatile int *modified1 = params.modified+1;  //modified flag for next/last iteration
    *modified1 = false;
    CudaDecoder::CostType cutoff=*params.cutoff;

    if (!is_init) {
        processEmittingTokens_function<32,2>(params);
        __grid_sync_nv_internal(params.barrier);  //ensure cur_toks size is final
    }
    int tok_E;
    int itv = params.verbose>2? 1: 10;
    if (rank0&&params.verbose>1&&params.frame%itv==0) 
      tok_E=params.cur_toks.size();

      *params.ne_idx=0;
      *params.l_ne_idx=params.cur_toks.size();
      int cnt=0;
      uint32_t size = 0;
      uint32_t psize=size;
    do {
      psize=size;
      size = params.cur_toks.size();
      if (rank0) {
        *params.ne_idx=0; //psize;
      }
      cnt++;
      bool aggregate=cnt>1?1:0;
      //grid.sync();  
      __grid_sync_nv_internal(params.barrier); //wait for everyone to read size and modified0

      //swap buffers
      swap(modified0,modified1); //double buffered to avoid extra sync when resetting modified to false

      *modified1 = false;

      processNonEmittingTokens_function<32,2>(params,cutoff,size,modified0, aggregate);

      //grid.sync();
      __grid_sync_nv_internal(params.barrier);  //wait for everyone to finish process tokens and writes modified0
    } while ((*modified0)==true);

    if (rank0&&params.verbose>1&&params.frame%itv==0) 
          printf("TK: %i %i %i %i\n", params.frame, tok_E, params.cur_toks.size(), cnt);

    allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks, params.allocator);
  
    if(rank0) {
      //prepare for next iteration
      params.prev_toks.clear();
      *params.cutoff = INFINITY;
      *params.fb_idx=0;  
      *params.pe_idx=0;
    }
    params.cur_toks.clear_sub();
    __grid_sync_nv_internal(params.barrier);  //wait for allocation to finish
    
    if(rank0) {
      params.allocator.advanceFront(params.cur_toks.size());
    }
    
  }

  void CudaDecoder::ProcessNonemitting() {
    nvtxRangePushA("ProcessNonemitting");
    // Processes nonemitting arcs for one frame.  Propagates within
    // cur_toks_.

    dim3 threads(32,1);

    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    processTokens_params params;

    initParams(params);

    processTokens_cg<<<blocks,threads,0,stream_comp>>>(params, true);

    cudaCheckError();
    nvtxRangePop();
  }


  void CudaDecoder::ProcessTokens() {
    nvtxRangePushA("ProcessTokens");
    processTokens_params params;
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    initParams(params);

     if (params.verbose>2&&params.frame==1) KALDI_LOG <<"# of blocks: "<<blocks.x<<std::endl;

     if (params.verbose>4) KALDI_LOG <<std::endl;
    cudaStreamWaitEvent(stream_comp,event_ll,0); //make sure log likelihoods are on the device before starting these kernels

     if (params.verbose>4)  KALDI_LOG <<std::endl;
#if 0
    void *args[] = { (void*) &params };
    cudaLaunchCooperativeKernel((void*)processTokens_cg, blocks, threads, args, 0, stream_comp);
#else
    processTokens_cg<<<blocks,threads,0,stream_comp>>>(params);  //doesn't work
#endif
    cudaCheckError();
      
    cudaEventSynchronize(event_pt); //throttle
    cudaEventRecord(event_pt,stream_comp);


    cur_toks_.copy_size_to_host(stream_comp); //for PreProcessTokens

    nvtxRangePop();
  }

} // end namespace kaldi.
