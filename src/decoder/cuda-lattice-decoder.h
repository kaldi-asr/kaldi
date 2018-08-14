// decoder/cuda-lattice-decoder.h

// Copyright      2018  Zhehuai Chen

// See ../../COPYING for clarification regarding multiple authors
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http:// www.apache.org/licenses/LICENSE-2.0
// 
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDA_LATTICE_DECODER_H_
#define KALDI_CUDA_LATTICE_DECODER_H_

#include "cuda-decoder-utils.h"

namespace kaldi {

#define LAT_BUF_SIZE 2
#define ESTIMATED_PRUNE_RATIO 0.25
class CudaLatticeDecoder;

struct CudaLatticeDecoderConfig {
  BaseFloat gpu_fraction;
  BaseFloat lat_fraction;
  uint32 max_tokens_per_frame;
  uint32 max_lat_arc_per_frame;
  uint32 max_tokens;
  uint32 max_arcs;
  BaseFloat lattice_beam;
  BaseFloat beam;
  uint32 max_len;
  int32 max_active;
  int32 chunk_len;
  BaseFloat acoustic_scale;

  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  bool determinize_lattice;
  int32 mem_print_freq;
  int32 verbose;
  
  CudaLatticeDecoderConfig(): 
                       gpu_fraction(1.0/8.0),
                       lat_fraction(1.0/2.0),
                       max_tokens_per_frame(800000),
                       max_lat_arc_per_frame(1000000),
                       max_tokens(15000000),
                       max_arcs(20000000), 
                       lattice_beam(10.0),
                       beam(16.0),
                       max_len(3000),
                       max_active(100000),
                       chunk_len(1),
                       acoustic_scale(1),
                       determinize_lattice(true),
                       mem_print_freq(10),
                       verbose(0) { }
 
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("cuda-verbose", &verbose, "debug log verbose.");
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("lat-fraction", &lat_fraction, 
      "Percent of GPU to use for lattice processing, i.e. gpu_fraction*lat_fraction");
    opts->Register("gpu-fraction", &gpu_fraction, 
      "Percent of GPU to use for this LatticeDecoder.  "
      "A single decoding cannot saturate the device.  "
      "Use multiple LatticeDecoders in parallel for the best performance.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, 
      "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-arcs-per-frame", &max_lat_arc_per_frame, 
      "Maximum arcs used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-tokens-allocated", &max_tokens, 
      "Total number of tokens allocated.  This controls how many tokens"
      " are allocated to the entire decoding process."
      "  If actual usaged exceeds this the results are undefined.");
    opts->Register("max-arcs-allocated", &max_arcs, 
      "Total number of arcs allocated.  This controls how many tokens " 
      " are allocated to the entire decoding process. "
      "  If actual usaged exceeds this the results are undefined.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("max-len", &max_len, "Decoder max len. ");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate. It's a faster but approximate version for GPU.");    
    opts->Register("chunk-len", &chunk_len, "chunk length for loading posteriors.");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && gpu_fraction>0 && gpu_fraction <= 1 &&
     lat_fraction>0 && lat_fraction <= 1 
      && max_tokens_per_frame > 0 && max_tokens>0 && lattice_beam > 0.0
                 && max_len > 0);
  }
};

class CudaLatticeDecoder {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef BaseFloat CostType;
  
  class LatticeProcessor;

  // general cuda vector can be used in both host and device. 
  // page faults need to be paid attention to
  // e.g. call CopySizeToHost() before Size() in host
  template<typename T>
  class CudaVector {
   public:
    void Allocate(uint32 max_size, 
       uint32* count_h=NULL, uint32* count_d=NULL, T* mem_d=NULL, T* mem_h=NULL) ;
    void Free(bool create_outside=false);

    HOST DEVICE T& operator[](uint32 idx); 
    HOST DEVICE const T& operator[](uint32 idx) const; 
    HOST DEVICE uint32 Size() const; 
    HOST DEVICE uint32 PushBack(const T &val); 
    HOST DEVICE void Clear(cudaStream_t stream=0); 
    bool Empty() const { return Size() == 0; }
    HOST DEVICE int32 GetIdxFromAddr(T* addr); 
    void Swap(CudaVector<T> &v); 
    size_t GetCudaMallocBytes() { return alloc_size; } 
    
    // a series of data transfer functions between host and device
    void CopyAllToHost(cudaStream_t stream=0);
    void CopyAllToDevice(cudaStream_t stream=0);
    void CopySizeToHost(cudaStream_t stream=0);
    void CopySizeToDevice(cudaStream_t stream=0);
    void CopyDataToHost(cudaStream_t stream=0, T* to_buf=NULL, 
                               bool copy_size=true);
    void CopyDataToDevice(cudaStream_t stream=0);
      
   protected:
    uint32 *count_d, *count_h; // current T number in buf
    uint32 max_size;  // buf size
    T* mem_d, *mem_h; // mem buf for T
    int32 alloc_size;
  };

  // complexer cuda vector used in 2-pass atomic token recombination
  template<typename T>
  class CudaMergeVector : public CudaVector<T> {
    friend class LatticeProcessor;
   
   public:
    using CudaVector<T>::operator[];
    using CudaVector<T>::PushBack;
    using CudaVector<T>::Clear;
    using CudaVector<T>::Size;

    void Allocate(uint32 max_size);
    void Free();   
    void Swap(CudaMergeVector<T> &v);
    size_t GetCudaMallocBytes();

    // according to the unpack index, copy data from external buf to the inside
    // buf; it's used in the 2nd stage of 2-pass atomic token recombination
    DEVICE void StoreDataByPackIdx(void* temp_data_buf, 
                      int* temp_data_buf_update, int32 buf_size,
                      LatticeProcessor *lattice_processor);
    // check whether data at index i is updated
    DEVICE int32 IsUpdated(int32 i);
    // push back data & data_pack to vectors respectively
    DEVICE uint32 PushBack(const T &val, uint64 *val_pack);  
  
   private:
    using CudaVector<T>::count_d;
    using CudaVector<T>::mem_d;
    using CudaVector<T>::max_size;

    // for arr merge to single; assume create using cudaMallocManaged
    int32 *mem_update_d;

    // recombination uint64 variable corresponding to each elem in T* mem_d
    // is stored in mem_d, and used in atomic based recombination 
    // _find_or_add_token_arc()

    int* barrier_;
  };

  // align to 16 bits so as to fast memcpy, see store16()
  class __align__(8) Token {
   public:
    CostType cost_; // accumulated total cost up to this place.
    BaseFloat extra_cost; // used in lattice pruning
    
    HOST DEVICE Token(BaseFloat cost, Token* prev) : 
                            cost_(cost), extra_cost(0) {
      assert(sizeof(Token)==8); 
      if(prev) {
        cost_ += prev->cost_;
      }
    }
    HOST DEVICE Token() { } 

    HOST DEVICE bool operator < (const Token &other) {
      return cost_ > other.cost_;
    }
    HOST DEVICE bool operator < (const Token &other) volatile{
      return cost_ > other.cost_;
    }
  };

  // we save all info in this structure, so as to collect all info together
  // in GPU memory and use memcpy to move to CPU memory
  // this structure is only used before D2H memcpy, during decoding, 
  // we use LatLinkCompact
  class __align__(32) LatLink {  
   public:
     // below variables are with the same size as ForwardLink, so as to enable memcpy
    void *p1; // pack of (int32 next_tok_id, int32 next_tok_fr;)
    int32 ilabel; // ilabel on link.
    int32 olabel; // olabel on link.
    BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    void *p2; // pack of (int32 prev_tok_id, int32 prev_tok_fr;)

    HOST DEVICE LatLink(int32 prev_tok_id, int32 prev_tok_fr,     
                               int32 next_tok_id, int32 next_tok_fr, 
        int32 ilabel, int32 olabel, BaseFloat graph_cost, BaseFloat acoustic_cost): 
        ilabel(ilabel), olabel(olabel), graph_cost(graph_cost), 
        acoustic_cost(acoustic_cost) {
      assert(sizeof(LatLink)==32); 
      p1=(void*)ENCODE_TOK_IDX_PAIR(next_tok_fr,next_tok_id);
      p2=(void*)ENCODE_TOK_IDX_PAIR(prev_tok_fr,prev_tok_id);
    }
  };

  // during decoding, as arcs are pre-allocated, we need a more compact 
  // structure and it is aligned in 16 bits. 
  // Notably, to work in 16 bits, we have to pack both id & whether it is 
  // emit arc in is_emit_pack_prev_tok_id
  class __align__(16) LatLinkCompact {  
   public:
    uint32 next_tok_id; // token index in the frame level token vector
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    int32 arc_id;

    HOST DEVICE LatLinkCompact(uint32 prev_tok_id, int32 prev_tok_fr,     
                                      uint32 next_tok_id, int32 next_tok_fr, 
                                      BaseFloat acoustic_cost, int32 arc_id): 
        next_tok_id(next_tok_id), 
        acoustic_cost(acoustic_cost), arc_id(arc_id),
        is_emit_pack_prev_tok_id(prev_tok_id) {
      assert(sizeof(LatLinkCompact)==16); 
      // we can't cope with that large number
      assert(is_emit_pack_prev_tok_id < ((uint32)1<<31));  
      uint32 is_emit_arc = prev_tok_fr != next_tok_fr;
      // a hack to save is_emit_arc in is_emit_pack_prev_tok_id
      this->is_emit_pack_prev_tok_id |= (is_emit_arc<<31); 
    }
    HOST DEVICE bool IsEmitArc() {
      return is_emit_pack_prev_tok_id >= ((uint32)1<<31);
    }
    HOST DEVICE uint32 GetPrevTokId() {
      return is_emit_pack_prev_tok_id & (((uint32)1<<31) - 1);
    }

   private:
    // a hack to contain both id & whether it is emit arc
    uint32 is_emit_pack_prev_tok_id;  
  };

  // align to 16 bits so as to fast memcpy, see store16()
  struct __align__(16) TokenState {
   public:

    volatile int32_t tok_idx_allocated; // this can ALSO be obtained from 
    //TokenLookupElem[state].tok_idx_allocated in each frame
    StateId state;  // record WFST state
    uint64 token_pack; // the real mem is here
    // if learned from TokenLookupElem that this TS is de-active, 
    // we need to firstly giving value or reset all the mem of this; 
    // thus don't need to call allocate_new_tokens
    
    HOST DEVICE TokenState (StateId state) : tok_idx_allocated(-1), state(state),
                            token_pack(pack_cost_idx_into_uint64(-FLT_MAX, 0)) {
      assert(sizeof(TokenState)==16); 
    }
  };

  // struct to hold pre-allocated tokens (one per WFST state) for fast lookup
  struct  TokenLookupElem{
    #define LOOKUP_STORE_TYPE uint32
    #define LOOKUP_DEACTIVE ((LOOKUP_STORE_TYPE)-1)
    #define LOOKUP_READY_PUSH ((LOOKUP_STORE_TYPE)-2)
    
    // used to index the corresponding TokenState
    volatile LOOKUP_STORE_TYPE tokenstate_idx; 
  };
  // typedef CudaVector<TokenState> TokenVector;
  typedef CudaMergeVector<TokenState> TokenMergeVector;
  typedef CudaVector<LatLinkCompact> LatLinkVector;


   // for lattice processing
  class LatticeProcessor {
   public:  
    void Initialize();
    int32 Allocate(int32 max_tokens_per_frame, int32 max_lat_arc_per_frame, 
      int32 max_len, int32 max_toks, int32 max_arcs, const CudaFst& fst);
    void Free();
    // The GPU memory of lattice arcs is shared with LatLinkVector
    LatLinkCompact* GetDeviceArcsBpr() { return arcs_bpr_d; } 

    DEVICE Token* GetTokenByExactIdx(uint32 offset);
    DEVICE int32 GetTokenIdxFromAddr(Token* tok);
    DEVICE int32 GetTokenAllocIdx(uint32 offset);

    // Entry of lattice pruning until this frame
    DEVICE void PruneActiveTokens(int32 frame, BaseFloat lattice_beam, int32 verbose);    
    // Collect after each token passing
    DEVICE void CollectToksPerFrame(TokenMergeVector& cur_toks_vec, 
                                            int32 frame);
    DEVICE void CollectArcsPerFrame(LatLinkVector& cur_arc_array,
                                             int32 frame);

    // Data transfer from device to host
    void CopyArcsToHost(int32 frame, cudaStream_t st);
    void CopyToksToHost(int32 frame, cudaStream_t st);
    void GetHostData(Token** toks_buf, int** toks_fr_sidx, 
                              LatLink** arcs_buf, int** arcs_fr_size);

   private:    
    // #define ENCODE_TOK_IDX_PAIR(frame,idx) (((uint64)(frame)<<32)+(idx))
    DEVICE int32 AddArc(LatLink* arc);
    DEVICE int32 AddArc(LatLinkCompact* arc, int32 frame);
    // Set start index in the buffer of the next frame
    DEVICE void SetNextSidx(int* sidx_buf, int32 size, int32 frame);
    DEVICE Token* GetActiveToken(void* p, bool check=false, int32 frame=-1) const;
    DEVICE Token* GetActiveToken(int32 frame, int32 id, bool check=false) const;
    DEVICE Token* GetActiveTokenByExactId(int32 frame, 
  int32 id_exact, bool check) const;

    DEVICE LatLinkCompact* GetActiveArc(int32 frame, int32 id) const;
    DEVICE int32 GetSize(int* acc_len, int32 frame) const;
    // used in PruneLatticeForFrame()
    DEVICE void UpdateModifiedFlags( 
                  volatile int32 **modified0, volatile int32** modified1,
                  volatile int32 **modified2, int cnt, int32* modified_d);
    
    // The parallel lattice pruning is based on the algorithm in
    // LatticeFasterDecoder::PruneActiveTokens 
    // with necessary modifications for GPU parallelization:
    // i) parallelize the iterative updating of nodes and arcs over GPU
    // threads; ii) use a global arc vector to replace the linked lists in
    // the old implementation, for its lack of random access features to
    // enable parallel access; iii) implement the extra cost updating as
    // an atomic operation to eliminate write conflicts among threads.
    // When a lattice arc is pruned, we do not physically remove
    // the arc, as memory allocation is expensive. Instead, we do a
    // final merging step to aggregate all remaining arcs using thread
    // parallelism 
    // We do not prune lattice nodes because: i) we need a static mapping
    // for each arc to trace the previous and the next nodes before
    // and after D2H memory copy. We use frame index t and vector
    // index i to trace a node, thus node positions in the vector cannot
    // be changed. ii) the lattice is constructed in CPU by iterating
    // remaining arcs, thus nodes are implicitly pruned. iii) node D2H copy is done
    // in each frame asynchronously, which does not introduce overheads.
    DEVICE void PruneLatticeForFrame(int32 frame, 
                  bool merge, BaseFloat lattice_beam, int32 verbose);

   private:
    // before pruning (bpr)

    // Preallocates tokens, allows threads to concurrently
    // allocate/deallocate objects quickly in GPU
    Token* toks_bpr_d; 
    Token* toks_bpr_h;
    // we keep start idx per-frame to fast index a token by (frame, idx) pair
    // see GetActiveToken() & GetActiveArc()
    int* toks_bpr_fr_sidx_d; 
    int* toks_bpr_fr_sidx_h;
    int* toks_num_used;
    // the GPU memory of lattice arcs is shared with LatLinkVector
    // see CudaLatticeDecoder::CudaLatticeDecoder()
    LatLinkCompact* arcs_bpr_d;
    // we keep start idx per-frame to fast index a arc by (frame, idx) pair
    int* arcs_bpr_fr_sidx_d; 
    int* arcs_bpr_used_d; // used in CollectArcsPerFrame() to get the size per-frame

    // after pruning (apr)
    // save size but not start idx because: i) it's in reverse order; 
    // of [frame-2*max_len+1, frame-1*max_len]
    // ii) we organize arcs by frame in CPU, which needs arc size per frame
    int* arcs_apr_fr_size_d; 
    int* arcs_apr_fr_size_h; 
    LatLink* arcs_apr_d;
    LatLink* arcs_apr_h;
    int* arcs_apr_used_d; // for atomic operations in mergeArc
    int* arcs_apr_used_h; // for final copying arcs to CPU

    // GPU global memory temp variables
    int32 *barrier_;
    int* count_vec_acc_d;
    int* modified_d;

    // configurations
    int32 max_len;
    int32 toks_buf_before_pr_size;
    int32 arcs_buf_before_pr_size;

    // for AddArc() from LatLinkCompact to LatLink
    // we record all information in LatLink to eliminate CPU memory lookup
    const int32 *arc_ilabels;
    const int32 *arc_olabels; 
    const BaseFloat *arc_weights;
  };
  
  // structs to hold kernel parameters.  Large numbers of parameters can slow down 
  // launch latency which matters when we are launching very short kernels 
  struct processTokens_params {
    // data
    TokenMergeVector prev_toks;
    TokenMergeVector cur_toks;
    TokenLookupElem *current_tokens_lookup;
    CostType *cutoff;
    CostType *cutoff_prev;
    LatLinkVector lat_arcs_sub_vec;
    int *lat_arcs_buf_end;
    Token* token_per_arc;
    int* token_per_arc_update;

    // tools
    LatticeProcessor lattice_processor;
    CudaHistogram histogram_prev_toks;

    // never change
    const __restrict__ uint32 *e_offsets;
    const __restrict__ uint32 *ne_offsets;
    const __restrict__ int32 *arc_ilabels;
    const __restrict__ int32 *arc_olabels; 
    const __restrict__ BaseFloat *arc_weights;
    const __restrict__ StateId *arc_nextstates;
    CuMatrixScaledMapper cuda_decodable;

    // GPU global memory temp variables
    volatile int32 *modified;
    int32 *ne_queue;
    int32 *barrier;
    int32 *num_arcs_till_last;

    // configurations
    BaseFloat beam;
    int32 verbose; // for debug 
    BaseFloat lattice_beam;
    int32 max_len;
    int32 numArcs;
    uint32 frame;   
    int32 max_lat_arc_per_frame;
    int max_active;

    // scan & expand
    int *d_q_token_from_narcs;
    int *d_block_sums_scan;
    int *d_n_CTA_done;
    int *d_block_sums;
    int *d_degrees_scan;
    int *d_q_arc_offset;
  };


 public:
  CudaLatticeDecoder(const CudaFst &fst, const TransitionModel &trans_model, 
                     const CudaLatticeDecoderConfig &config);  
  ~CudaLatticeDecoder();

  // decoding functions
  void InitParams(processTokens_params* params);  // parameters for calling GPU
  // call InitDecoding if you have already decoded an
  // utterance and want to start with a new utterance. 
  void InitDecoding(); 
  void UpdateTokPointersByFrame(uint32 frame);
  // Returns the number of frames already decoded.  
  int32 NumFramesDecoded() const { return num_frames_decoded_; }
  void ClearToks(TokenMergeVector &toks); 
  void PreProcessTokens();  // called before ProcessTokens()
  // ProcessTokens decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  void ProcessTokens();
  void ProcessNonemitting(); // only called at frame 0
  void DecodeChunk(CuMatrix<BaseFloat>* post_chunk);
  void Decode(MatrixChunker *decodable);

  // lattice processing functions
  void FinalProcessLattice(Token** toks_buf, int** toks_fr_sidx, 
                                 LatLink** arcs_buf, int** arcs_fr_size,
                                 TokenMergeVector** toks_vec_last_fr, int *num_frames_decoded);
  void PruneActiveTokens(cudaStream_t wait_st, cudaStream_t run_st, 
      BaseFloat gpu_ratio); // prune lattice in GPU

  // other functions
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;
  size_t GetCudaMallocBytes() const { return bytes_cuda_malloc; } 
  size_t GetCudaMallocManagedBytes() const { return bytes_cuda_malloc_managed;  }

 private:
  // configurations
  CudaLatticeDecoderConfig config_;
  const CudaFst fst_;

  int32 *ne_queue_d; // for token idx
  // token passing
  TokenMergeVector* cur_toks_;
  TokenMergeVector* prev_toks_;  
  CostType *cutoff_d;
  CostType *cutoff_prev_d;
  int32 *num_arcs_till_last_d;
  int32 *modified_d; // used in processTokens_cg()
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;
  // 2-stage atomic token recombination
  Token* token_per_arc_d; // token array whose length equal to size of WFST arcs
  int32 *token_per_arc_update_d; // to check whether the token is updated at this frame  
  // token lookup table.  Provides constant time lookup for active tokens.
  // One entry per state. TokenLookupElem::active to denote whether it is active.
  TokenLookupElem *current_tokens_lookup_d;
  CudaHistogram histogram_prev_toks_;

  // lattice
  TokenMergeVector lat_toks_bufs_[LAT_BUF_SIZE];
  LatLinkVector lat_arcs_buf_;
  int32* lat_arcs_buf_end_;
  LatticeProcessor lattice_processor_;

  // scan & expand
  int32 *d_q_token_from_narcs;
  int32 *d_block_sums_scan;
  int32 *d_n_CTA_done;
  int32 *d_block_sums;
  int32 *d_degrees_scan;
  int32 *d_q_arc_offset;

  // data store for log likelihoods needed in the current frame.  
  CuMatrixScaledMapper cuda_decodable_;
  const TransitionModel &trans_model_;  // for tid to pdf mapping
  int32* id2pdf_d_;

  // GPU usage
  uint32 total_threads; // GPU utilization
  size_t bytes_cuda_malloc, bytes_cuda_malloc_managed;
  int32 *barrier_d;  // barrier to allow grid syncs
  cudaEvent_t event_pt; // token passing
  cudaEvent_t event_ll; // log likelihoods calculation
  cudaStream_t stream_comp; // decoding
  cudaStream_t stream_lat[LAT_BUF_SIZE]; // lattice processing and copying
  cudaStream_t stream_ll; // log likelihoods calculation

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaLatticeDecoder);
};

} // end namespace kaldi.


#endif
