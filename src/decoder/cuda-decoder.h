// decoder/cuda-decoder.h

// Copyright      2018  Zhehuai Chen; Hugo Braun; Justin Luitjens

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

#ifndef KALDI_DECODER_CUDA_DECODER_H_
#define KALDI_DECODER_CUDA_DECODER_H_

#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "cuda-decoder-utils.h"

namespace kaldi {

class CudaDecoder;

struct CudaDecoderConfig {
  BaseFloat beam;
  double gpu_fraction;
  uint32 max_tokens;
  int32 max_active;
  uint32 max_tokens_per_frame;
  int32 max_len;
  BaseFloat acoustic_scale;
  int32 chunk_len;

  CudaDecoderConfig(): beam(16.0),
    gpu_fraction(1.0 / 8.0),
    max_tokens(300000000),
    max_active(100000),
    max_tokens_per_frame(1000000),
    max_len(50000),
    acoustic_scale(0.1), chunk_len(1) {}

  void Register(OptionsItf *opts) {
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("gpu-fraction", &gpu_fraction,
                   "Percent of GPU to use for this decoder.  "
                   "A single decoding cannot saturate the device.  "
                   "Use multiple decoders in parallel for the best performance.");
    opts->Register("max-tokens-allocated", &max_tokens,
                   "Total number of tokens allocated.  This controls how many tokens "
                   "are allocated to the entire decoding process."
                   "  If actual usaged exceeds this the results are undefined.");
    opts->Register("max-active", &max_active,
                   "Decoder max active states.  Larger->slower; "
                   "more accurate. It's a faster but approximate version for GPU.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, 
      "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-len", &max_len, "Decoder max len. ");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
    opts->Register("chunk-len", &chunk_len, "chunk length for loading posteriors.");

  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && gpu_fraction > 0 && gpu_fraction <= 1
                 && max_tokens_per_frame > 0 && max_tokens > 0 && chunk_len > 0);
  }
};

class CudaDecoder {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef float CostType;

  struct Token {
    // to mostly read in coalesced accesses, we needed to take StateId out
    BaseFloat cost; // accumulated total cost up to this point.
    int prev_token;
    int arc_idx;
  };

  struct processTokens_params {
    CuMatrixScaledMapper cuda_decodable;
    CudaHistogram histogram_prev_toks;

    StateId *d_q;
    Token *d_q_info;

    int *tok_from_d_;
    int *tok_to_d_;
    int *tok_end_d_;
    int *tot_narcs_d_;
    int *tot_narcs_h_;
    int *arc_offset_pertok_d_;
    int *tot_ntok_h_; // to be set at the end
    int *cur_tok_from_d_;
    uint *arc_offset_d;
    int *narcs_scan_d_;
    int* narcs_blksum_scan_d_;
    uint64 *d_lookup;

    BaseFloat *cutoff_d_;
    BaseFloat *cutoff_prev;

    int *arc_ilabels;
    BaseFloat *arc_weights;
    StateId *arc_nextstates;

    int max_active;
    BaseFloat beam;
    bool is_emitting;
    int frame;
    int *active_tok_cnt_d;

    int *n_CTA_d_;
    int *barrier;
  };

  CudaDecoder(const CudaFst &fst, const TransitionModel &trans_model,
              const CudaDecoderConfig &config);
  ~CudaDecoder();

  void InitDecoding();
  void Decode(MatrixChunker *decodable); // Decode an utterance of speech
  void DecodeChunk(CuMatrix<BaseFloat> *post_chunk); // Decode a chunk of speech

  int32 NumFramesDecoded() const { return num_frames_decoded_; }
  bool ReachedFinal() const;
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;
  BaseFloat FinalRelativeCost() const;

 private:
  void InitLookup();

  bool ProcessToken(bool is_emitting);
  void ProcessEmitting();
  void ProcessNonemitting();

  // functions called by ProcessToken
  void InitParams(processTokens_params* params, bool is_emitting);
  // Compute degrees, reduce by key, apply cutoff
  // Compute first part of the prefix sums of the degrees
  // At the end of that step, the kernel
  // set the value of tot_narcs_h_
  // (the number of arcs in the current queue processed)
  // The detailed description can be referred to GPU kernel definition
  void ContractAndPreprocess(const processTokens_params &params);
  // The description can be referred to GPU kernel definition
  void ExpandArcs(int nthreads, const processTokens_params &params);
  // a combination of above two processes with a single kernel
  void NonEmittingLongTail(const processTokens_params &params);

  void GetBestCost(BaseFloat *min, int *arg, bool isfinal) const;

  const CudaDecoderConfig &config_;
  const CudaFst fst_;
  // for tid to pdf mapping
  const TransitionModel &trans_model_;
  int32* id2pdf_d_;
  CuMatrixScaledMapper cuda_decodable_; // keep loglikelihoods
  CudaHistogram histogram_prev_toks_;

  // cutoff of current frame
  BaseFloat *cutoff_d_;
  // cutoff of prev frame
  BaseFloat *cutoff_prev_d_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;
  int *active_tok_cnt_d_;
  // Lookup table of each WFST state, it is a pack of [score, token_idx]
  uint64 *state_pack_d_;
  // store info of tokens
  StateId *token_stateid_d_;
  Token *token_stateid_d_Info;

  // used in Compute degrees
  // Total number of arcs from tok.next_state
  int *tot_narcs_d_, *tot_narcs_h_;
  // number of tokens in the queue
  int *tot_ntok_h_;
  // Scan of the outgoing arc degrees of tokens
  int *narcs_scan_d_;
  // number of arcs in the corresponding CTA block
  int *narcs_blksum_scan_d_;
  // the starting arc_id of out-going arcs from each token
  int *arc_offset_pertok_d_;

  // Save the offset of currToken of the current frame
  int *cur_tok_from_d_; // Used for ProcessEmitting of following frame
  // At each ProcessToken, we will propagate the token queue
  int *tok_from_d_;
  int *tok_to_d_;
  int *tok_end_d_;

  // backtrack the path
  int *reached_final_h_;
  StateId *reversed_path_d_, *reversed_path_h_;
  int *path_size_d_;

  // Used to detect last CTA alive in some kernels
  int *n_CTA_d_;
  int *barrier_d_; // for grid sync
  // Streams, overlap loglikelihoods copies with compute
  cudaStream_t stream_comp, stream_ll;
  cudaEvent_t event_ll; // finish loglikelihoods copy

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
};


} // end namespace kaldi.


#endif
