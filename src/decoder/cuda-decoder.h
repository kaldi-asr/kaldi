// decoder/cuda-decoder.h

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

#ifndef KALDI_CUDA_DECODER_H_
#define KALDI_CUDA_DECODER_H_



namespace kaldi {
  
class CudaDecoder;



struct CudaDecoderConfig {
  BaseFloat beam;
  double gpu_fraction;
  uint32_t max_tokens_per_frame;
  uint32_t max_tokens;
  int verbose;
  uint32_t max_lat_arc_per_frame;
  
  CudaDecoderConfig(): beam(16.0),
                       gpu_fraction(1.0/8.0),
                       max_tokens_per_frame(1<<17),
                       max_tokens(60000000),
                       verbose(0),
                       max_lat_arc_per_frame(1<<18)
                       {}
  
  void Register(OptionsItf *opts) {
    opts->Register("verbose", &verbose, "debug log verbose.");
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("gpu-fraction", &gpu_fraction, "Percent of GPU to use for this decoder.  "
                                                  "A single decoding cannot saturate the device.  "
                                                  "Use multiple decoders in parallel for the best performance.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-tokens-allocated", &max_tokens, "Total number of tokens allocated.  This controls how many tokens are allocated to the entire decoding process."
                                                        "  If actual usaged exceeds this the results are undefined.");
    opts->Register("max-lat-arc-per-frame", &max_lat_arc_per_frame, "Total number of lat arc allocated.  ");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && gpu_fraction>0 && gpu_fraction <= 1 && max_tokens_per_frame > 0 && max_tokens>0);
  }
};


class CudaDecoder {

  template<typename T>
class CudaVector {
    public:
     inline HOST DEVICE T& operator[](uint32_t idx); 
     inline HOST DEVICE const T& operator[](uint32_t idx) const; 
     inline void allocate(uint32_t max_size, 
        uint32_t* icount_h=NULL, uint32_t* icount_d=NULL, T* mem_d=NULL, T* mem_h=NULL) ;
     inline void free(bool create_outside=false);
     inline HOST DEVICE uint32_t size() const; 
      HOST DEVICE inline uint32_t push_back(const T &val); 
      HOST DEVICE inline void clear(cudaStream_t stream=0); 
      HOST DEVICE inline int get_idx_from_addr(T* addr); 
      inline bool empty() const;
      inline void swap(CudaVector<T> &v); 
      inline void copy_all_to_host(cudaStream_t stream=0);
      inline void copy_all_to_device(cudaStream_t stream=0);
      inline void copy_size_to_host(cudaStream_t stream=0);
      inline void copy_size_to_device(cudaStream_t stream=0);
      inline void copy_data_to_host(cudaStream_t stream=0, T* to_buf=NULL, bool copy_size=true);
      inline void copy_data_to_device(cudaStream_t stream=0);
      inline void copy_data_to_device(int size, T* mem_in_d, cudaStream_t stream=0);

      inline size_t getCudaMallocBytes(); 
      
    public:
      uint32_t *count_d, *count_h;
      uint32_t max_size;
      T* mem_d, *mem_h;
      int alloc_size;
};


template<typename T>
class CudaMergeVector : public CudaVector<T> {
public:
  using CudaVector<T>::operator[];
  using CudaVector<T>::push_back;
  using CudaVector<T>::size;
  using CudaVector<T>::count_d;
  using CudaVector<T>::mem_d;
  using CudaVector<T>::max_size;
  
  DEVICE inline void merge(void* undefined, int* token_per_arc_update, int num_arcs,  bool clear=true);
  DEVICE inline int update(int i);
  DEVICE inline void clear_sub();
  inline void allocate(uint32_t max_size);
  DEVICE inline uint32_t push_back(const T &val, uint64 *val_pack); 
  inline void free();
  inline size_t getCudaMallocBytes(); 
  inline void swap(CudaMergeVector<T> &v);

  //for arr merge to single; assume create using cudaMallocManaged
  int *mem_update_d;
  uint64** mem_pack_buf_d;
  T* mem_buf_d;
  int *mem_buf_count_d;
  int *mem_buf_acc_count_d;
  int* barrier_;
};
 
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef float CostType;

  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config);  
  ~CudaDecoder();

  inline size_t getCudaMallocBytes() const { return bytes_cudaMalloc; } 
  inline size_t getCudaMallocManagedBytes() const { return bytes_cudaMallocManaged;  }

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Decode(DecodableInterface *decodable);

  bool ReachedFinal() const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account final-probs.
  // fst_out will be empty (Start() == kNoStateId) if nothing was available due to
  // search error.
  // If Decode() returned true, it is safe to assume GetBestPath will return true.
  // It returns true if the output lattice was nonempty (i.e. had states in it);
  // using the return value is deprecated.
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;
  
  /// *** The next functions are from the "new interface". ***
  
  /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost plus
  /// cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were present
  /// on the final frame.  It will usually be nonnegative.
  BaseFloat FinalRelativeCost() const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance. 
  void InitDecoding();  

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  
  /// Returns the number of frames already decoded.  
  int32 NumFramesDecoded() const { return num_frames_decoded_; }


 
  class __align__(16) Token {
   public:
    Token *prev_;
    CostType cost_; // accumulated total cost up to this point.
    uint32_t arc_index_;
//    BaseFloat acoustic_cost;   //currently not recording acoustic_cost.  It is trivial to add back in but didn't seem necessary for this use case

    HOST DEVICE inline Token(BaseFloat cost, Token *prev, uint32_t arc_index) : prev_(prev), cost_(cost), arc_index_(arc_index) {
      //assert(prev!=this);
      if(prev) {
        cost_ += prev->cost_;
      }
    }

    HOST DEVICE inline bool operator < (const Token &other) {
      return cost_ > other.cost_;
    }
    HOST DEVICE inline bool operator < (const Token &other) volatile{
      return cost_ > other.cost_;
    }
  };
  
  //Preallocates tokens and allocates them in a circular buffer.
  //This allows threads to concurrently allocate/deallocate objects quickly in CUDA
  class TokenAllocator {
    public:
      void initialize(uint32_t size);
      void finalize();

      inline void prefetch_next_to_device(cudaStream_t stream, int count);
      inline void prefetch_next_to_device(cudaStream_t stream);
      inline void prefetch_allocated_to_host(cudaStream_t stream);

      inline size_t getCudaMallocManagedBytes();

      //circular buffer,  need to ensure front never gets close to back....  If this happens there can be race conditions 

      DEVICE inline Token* getToken(uint32_t index);   //gets a free token offset by index
      DEVICE inline void advanceFront(uint32_t num);         //advances the allocated token list by num

      void reset();   //returns all memory to the allocator (essentially a garbage collection of oustanding memory.  
    private:

      uint32_t size;
      int32_t device;
      uint32_t *front_d, *front_h;    //next free token index

      Token *tokens_allocation;  //TODO we could have a list of these and dynamically add more.  Just going static for now.
      size_t bytes_cudaMallocManaged;
      uint32_t prefetch_size;         //amount of elements to prefetch beyond front
  };

  TokenAllocator allocator;

  //pre-computes log likelihoods for the current frame
  void ComputeLogLikelihoods(DecodableInterface *decodable);
 
  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  //void ProcessEmitting(DecodableInterface *decodable);
  //

  struct TokenState;
  struct  TokenLookupElem;
  typedef CudaVector<TokenState> TokenVector;
  typedef CudaMergeVector<TokenState> TokenMergeVector;
  struct processTokens_params {

    CudaDecoder::TokenMergeVector prev_toks;
    CudaDecoder::TokenMergeVector cur_toks;
    CudaDecoder::TokenAllocator allocator;
    CudaDecoder::CostType *cutoff;

    //never change
    const __restrict__ uint32_t *e_offsets;
    const __restrict__ uint32_t *ne_offsets;
    const __restrict__ int32 *arc_ilabels;
    const __restrict__ int32 *arc_olabels; 
    const __restrict__ BaseFloat *arc_weights;
    const __restrict__ CudaDecoder::StateId *arc_nextstates;
    const __restrict__ BaseFloat *loglikelihoods;
    CudaDecoder::TokenLookupElem *current_tokens_lookup;
    volatile int *token_locks;
    BaseFloat beam;
    volatile int *modified;
    int *pe_idx;
    int *ne_idx;
    int *ne_queue;
    int *l_ne_idx;
    int *fb_idx;
    int *cidx2;
    int *cidx;
    int *barrier;

    //debug
    int verbose;
    int frame;

    int *tok2scansum_numarc;
    int *tid2arc;
    int *tid2tok;
    int max_arcs_per_frame_search;
    uint64 *clock_buf;
    Token* token_per_arc;
    int* token_per_arc_update;
    int numArcs;
  };


  void ProcessNonemitting();
  void ProcessTokens();
  void PreProcessTokens();
  void initParams(processTokens_params& params);
 
  //struct to hold pre-allocated tokens (one per state)
  struct  TokenLookupElem{
    Token *token;     //pointer for that token
    uint32_t active;  //tells if token has activiated or not
    uint64_t token_pack;     //aligning to 16 bytes
  };
  
  //token lookup table.  Provides constant time lookup for active tokens.
  //One entry per state.  If entry is NULL token is not active.
  TokenLookupElem *current_tokens_lookup_d;

  struct __align__(16) TokenState {
    Token* token;
    StateId state;
    int pad;
    HOST DEVICE inline TokenState (Token *token, StateId state) : token(token), state(state), pad(0) {}
    HOST DEVICE inline TokenState () : token(NULL) {};
  };
 

  //Lists of active tokens to be iterated through
  TokenMergeVector cur_toks_;
  TokenMergeVector prev_toks_;
  Token* token_per_arc_d;
  int *token_per_arc_update_d;

  int* tid2arc_d;
  int* tid2tok_d;
  int* tok2scansum_numarc_d;
  void* d_temp_storage;
  size_t temp_storage_bytes;
  uint64* clock_buf_d;

  const CudaFst fst_;

  BaseFloat beam_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  //data store for log likelihoods needed in the current frame.  Double buffering to avoid synchronization.
  BaseFloat *loglikelihoods_h, *loglikelihoods_old_h, *loglikelihoods_d, *loglikelihoods_old_d;  

  CostType *cutoff_d;
  int *modified_d;

  volatile int *token_locks_d;
  void ClearToks(TokenMergeVector &toks);

  cudaEvent_t event_pt, event_pt_old, event_ll;
  cudaStream_t stream_comp, stream_copy, stream_ll;

  uint32_t total_threads;
  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  //warp assignment indexes
  int *pe_idx_d, *ne_idx_d, *fb_idx_d, *l_ne_idx_d, *ne_queue_d;
  int *barrier_d;  //barrier to allow grid syncs
 
  int *cidx_d,*cidx2_d; //for less NE proc
  int verbose;
  
  int max_arcs_per_frame_search_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
};


} // end namespace kaldi.


#endif
