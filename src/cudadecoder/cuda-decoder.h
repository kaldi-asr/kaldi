// cudadecoder/cuda-decoder.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDADECODER_CUDA_DECODER_H_
#define KALDI_CUDADECODER_CUDA_DECODER_H_

#if HAVE_CUDA

#include <cuda_runtime_api.h>

#include <atomic>
#include <cfloat>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stack>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cudadecoder/cuda-decodable-itf.h"
#include "cudadecoder/cuda-decoder-common.h"
#include "cudadecoder/cuda-fst.h"
#include "cudadecoder/thread-pool-light.h"
#include "fst/symbol-table.h"
#include "online2/online-endpoint.h"

namespace kaldi {
namespace cuda_decoder {

struct CudaDecoderConfig {
  BaseFloat default_beam;
  BaseFloat lattice_beam;
  int32 ntokens_pre_allocated;
  int32 main_q_capacity, aux_q_capacity;
  int32 max_active;
  OnlineEndpointConfig endpointing_config;

  CudaDecoderConfig()
      : default_beam(15.0),
        lattice_beam(10.0),
        ntokens_pre_allocated(1000000),
        main_q_capacity(-1),
        aux_q_capacity(-1),
        max_active(10000) {}

  void Register(OptionsItf *opts) {
    opts->Register("beam", &default_beam,
                   "Decoding beam. Larger->slower, more accurate. If "
                   "aux-q-capacity is too small, we may decrease the beam "
                   "dynamically to avoid overflow (adaptive beam, see "
                   "aux-q-capacity parameter)");
    opts->Register("lattice-beam", &lattice_beam,
                   "The width of the lattice beam");
    opts->Register("max-active", &max_active,
                   "At the end of each frame computation, we keep only its "
                   "best max-active tokens. One token is the instantiation of "
                   "a single arc. Typical values are within the 5k-10k "
                   "range.");
    opts->Register("ntokens-pre-allocated", &ntokens_pre_allocated,
                   "Advanced - Number of tokens pre-allocated in host "
                   "buffers. "
                   "If this size is exceeded the buffer will reallocate, "
                   "reducing performance.");
    std::ostringstream main_q_capacity_desc;
    main_q_capacity_desc
        << "Advanced - Capacity of the main queue : Maximum number "
           "of "
           "tokens that can be stored *after* pruning for each "
           "frame. "
           "Lower -> less memory usage, Higher -> More accurate. "
           "Tokens stored in the main queue were already selected "
           "through a max-active pre-selection. It means that for "
           "each "
           "emitting/non-emitting iteration, we can add at most "
           "~max-active tokens to the main queue. Typically only "
           "the "
           "emitting iteration creates a large number of tokens. "
           "Using "
           "main-q-capacity=k*max-active with k=4..10 should be "
           "safe. "
           "If main-q-capacity is too small, we will print a "
           "warning "
           "but prevent the overflow. The computation can safely "
           "continue, but the quality of the output may decrease "
           "(-1 = set to "
        << KALDI_CUDA_DECODER_MAX_ACTIVE_MAIN_Q_CAPACITY_FACTOR
        << "*max-active).";
    opts->Register("main-q-capacity", &main_q_capacity,
                   main_q_capacity_desc.str());
    std::ostringstream aux_q_capacity_desc;
    aux_q_capacity_desc
        << "Advanced - Capacity of the auxiliary queue : Maximum "
           "number of raw tokens that can be stored *before* "
           "pruning "
           "for each frame. Lower -> less memory usage, Higher -> "
           "More "
           "accurate. During the tokens generation, if we detect "
           "that "
           "we are getting close to saturating that capacity, we "
           "will "
           "reduce the beam dynamically (adaptive beam) to keep "
           "only "
           "the best tokens in the remaining space. If the aux "
           "queue "
           "is still too small, we will print an overflow warning, "
           "but "
           "prevent the overflow. The computation can safely "
           "continue, "
           "but the quality of the output may decrease. We "
           "strongly "
           "recommend keeping aux-q-capacity large (>400k), to "
           "avoid "
           "triggering the adaptive beam and/or the overflow "
           "(-1 = set to "
        << KALDI_CUDA_DECODER_AUX_Q_MAIN_Q_CAPACITIES_FACTOR
        << "*main-q-capacity).";
    opts->Register("aux-q-capacity", &aux_q_capacity,
                   aux_q_capacity_desc.str());
    endpointing_config.Register(opts);
  }

  void Check() const {
    KALDI_ASSERT(default_beam > 0.0 && ntokens_pre_allocated >= 0 &&
                 lattice_beam >= 0.0f && max_active > 0);
  }

  void ComputeConfig() {
    if (main_q_capacity == -1)
      main_q_capacity =
          max_active * KALDI_CUDA_DECODER_MAX_ACTIVE_MAIN_Q_CAPACITY_FACTOR;
    if (aux_q_capacity == -1)
      aux_q_capacity =
          main_q_capacity * KALDI_CUDA_DECODER_AUX_Q_MAIN_Q_CAPACITIES_FACTOR;
  }
};

// Forward declaration.
// Those contains CUDA code. We don't want to include their definition
// in this header
struct DeviceParams;
struct KernelParams;

class CudaDecoder {
 public:
  // Creating a new CudaDecoder, associated to the FST fst
  // nlanes and nchannels are defined as follow

  // A decoder channel is linked to one utterance.
  // When we need to perform decoding on an utterance,
  // we pick an available channel, call InitDecoding on that channel
  // (with that ChannelId in the channels vector in the arguments)
  // then call AdvanceDecoding whenever frames are ready for the decoder
  // for that utterance (also passing the same ChannelId to
  // AdvanceDecoding)
  //
  // A decoder lane is where the computation actually happens
  // a decoder lane is channel, and perform the actual decoding
  // of that channel.
  // If we have 200 lanes, we can compute 200 utterances (channels)
  // at the same time. We need many lanes in parallel to saturate the big
  // GPUs
  //
  // An analogy would be lane -> a CPU core, channel -> a software thread
  // A channel saves the current state of the decoding for a given
  // utterance. It can be kept idle until more frames are ready to be
  // processed
  //
  // We will use as many lanes as necessary to saturate the GPU, but not
  // more. A lane has an higher memory usage than a channel. If you just
  // want to be able to keep more audio channels open at the same time
  // (when I/O is the bottleneck for instance, typically in the context of
  // online decoding), you should instead use more channels.
  //
  // A channel is typically way smaller in term of memory usage, and can
  // be used to oversubsribe lanes in the context of online decoding For
  // instance, we could choose nlanes=200 because it gives us good
  // performance
  // on a given GPU. It gives us an end-to-end performance of 3000 XRTF.
  // We are doing online, so we only get audio at realtime speed for a
  // given utterance/channel. We then decide to receive audio from 2500
  // audio channels at the same time (each at realtime speed), and as soon
  // as we have frames ready for nlanes=200 channels, we call
  // AdvanceDecoding on those channels
  // In that configuration, we have nlanes=200 (for performance), and
  // nchannels=2500 (to have enough audio
  // available at a given time).
  // Using nlanes=2500 in that configuration would first not be possible
  // (out of memory), but also not necessary. Increasing the number of
  // lanes is only useful if it increases performance. If the GPU is
  // saturated at nlanes=200, you should not increase that number.
  //
  ///\param[in] fst A CudaFst instance. Not owned, must survive this object.
  ///\param[in] config
  ///\param[in] nlanes
  ///\param[in] nchannels
  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config, int32 nlanes,
              int32 nchannels);

  // Special constructor for nlanes = nchannels. Here for the non-advanced
  // user Here we can consider nchannels = batch size. If we want to
  // decode 10 utterances at a time, we can use nchannels = 10
  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config,
              int32 nchannels)
      : CudaDecoder(fst, config, nchannels, nchannels) {}
  virtual ~CudaDecoder() noexcept(false);

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);

  // Reads the config from config
  void ReadConfig(const CudaDecoderConfig &config);
  // InitDecoding initializes the decoding, and should only be used if you
  // intend to call AdvanceDecoding() on the channels listed in channels
  void InitDecoding(const std::vector<ChannelId> &channels);
  // Computes the heavy H2H copies of InitDecoding. Usually launched on
  // the threadpool
  void InitDecodingH2HCopies(ChannelId ichannel);
  // AdvanceDecoding on a given batch
  // a batch is defined by the channels vector
  // We can compute N channels at the same time (in the same batch)
  // where N = number of lanes, as defined in the constructor
  // AdvanceDecoding will compute as many frames as possible while running
  // the full batch when at least one channel has no more frames ready to
  // be computed, AdvanceDecoding returns The user then decides what to
  // do, i.e.:
  //
  // 1) either remove the empty channel from the channels list
  // and call again AdvanceDecoding
  // 2) or swap the empty channel with another one that has frames ready
  // and call again AdvanceDecoding
  //
  // Solution 2) should be preferred because we need to run full, big
  // batches to saturate the GPU
  //
  // If max_num_frames is >= 0 it will decode no more than
  // that many frames.
  void AdvanceDecoding(
      const std::vector<std::pair<ChannelId, const BaseFloat *>> &lanes_assignements);

  // Version with deprecated API - will be removed at some point
  void AdvanceDecoding(const std::vector<ChannelId> &channels,
                       std::vector<CudaDecodableInterface *> &decodables,
                       int32 max_num_frames = -1);

  void AllowPartialHypotheses() {
    partial_traceback_ = generate_partial_hypotheses_ = true;
  }

  void AllowEndpointing() {
    if (frame_shift_seconds_ == FLT_MAX) {
      KALDI_ERR << "You must call SetOutputFrameShiftInSeconds() "
                << "to use endpointing";
    }
    partial_traceback_ = endpointing_ = true;
  }

  void SetOutputFrameShiftInSeconds(BaseFloat f) { frame_shift_seconds_ = f; }

  void GetPartialHypothesis(ChannelId ichannel, PartialHypothesis **out) {
    KALDI_ASSERT(generate_partial_hypotheses_);
    // No need to lock, all ops on h_all_channels_partial_hypotheses_out_ are
    // done before returning InitDecoding or AdvanceDecoding
    *out = &h_all_channels_partial_hypotheses_out_[ichannel];
  }

  bool EndpointDetected(ChannelId ichannel) {
    return h_all_channels_endpoint_detected_[ichannel];
  }

  // Returns the number of frames already decoded in a given channel
  int32 NumFramesDecoded(ChannelId ichannel) const;

  // GetBestPath gets the one-best decoding traceback. If
  // "use_final_probs" is true AND we reached a final state, it limits
  // itself to final states; otherwise it gets the most likely token not
  // taking into account final-probs.
  // GetBestPath is deprecated and will be removed in a future release
  // For best path, use partial hypotheses
  void GetBestPath(const std::vector<ChannelId> &channels,
                   std::vector<Lattice *> &fst_out_vec,
                   bool use_final_probs = true);
  // It is possible to use a threadsafe version of GetRawLattice, which is
  // ConcurrentGetRawLatticeSingleChannel()
  // Which will do the heavy CPU work associated with GetRawLattice
  // It is necessary to first call PrepareForGetRawLattice *on the main
  // thread* on the channels. The main thread is the one we use to call
  // all other functions, like InitDecoding or AdvanceDecoding We usually
  // call it "cuda control thread", but it is a CPU thread For example: on
  // main cpu thread : Call PrepareForGetRawLattice on channel 8,6,3 then:
  // on some cpu thread : Call ConcurrentGetRawLatticeSingleChannel on
  // channel 3 on some cpu thread : Call
  // ConcurrentGetRawLatticeSingleChannel on channel 8 on some cpu thread
  // : Call ConcurrentGetRawLatticeSingleChannel on channel 6
  void PrepareForGetRawLattice(const std::vector<ChannelId> &channels,
                               bool use_final_probs);
  void ConcurrentGetRawLatticeSingleChannel(ChannelId ichannel,
                                            Lattice *fst_out);

  // GetRawLattice gets the lattice decoding traceback (using the
  // lattice-beam in the CudaConfig parameters). If "use_final_probs" is
  // true AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account
  // final-probs.
  void GetRawLattice(const std::vector<ChannelId> &channels,
                     std::vector<Lattice *> &fst_out_vec, bool use_final_probs);

  // (optional) Giving the decoder access to the cpu thread pool
  // We will use it to compute specific CPU work, such as
  // InitDecodingH2HCopies For recurrent CPU work, such as
  // ComputeH2HCopies, we will use dedicated CPU threads We will launch
  // nworkers of those threads
  void SetThreadPoolAndStartCPUWorkers(ThreadPoolLight *thread_pool,
                                       int32 nworkers);

  // Used to generate partial results
  void SetSymbolTable(const fst::SymbolTable &word_syms) {
    word_syms_ = &word_syms;
  }

 private:
  // Data allocation. Called in constructor
  void AllocateDeviceData();
  void AllocateHostData();
  void AllocateDeviceKernelParams();
  // Data initialization. Called in constructor
  void InitDeviceData();
  void InitHostData();
  void InitDeviceParams();
  // Computes the initial channel
  // The initial channel is used to initialize a channel
  // when a new utterance starts (we clone it into the given channel)
  void ComputeInitialChannel();
  // Updates *h_kernel_params using channels
  void SetChannelsInKernelParams(const std::vector<ChannelId> &channels);
  void ResetChannelsInKernelParams();
  // Context-switch functions
  // Used to perform the context-switch of load/saving the state of a
  // channels into a lane. When a channel will be executed on a lane, we
  // load that channel into that lane (same idea than when we load a
  // software threads into the registers of a CPU)
  void LoadChannelsStateToLanes(const std::vector<ChannelId> &channels);
  void SaveChannelsStateFromLanes();
  // GetBestCost finds the best cost in the last tokens queue
  // for each channel in channels. If isfinal is true,
  // we also add the final cost to the token costs before
  // finding the minimum cost
  // We list all tokens that have a cost within [best; best+lattice_beam]
  // in list_lattice_tokens.
  // We alsos set has_reached_final[ichannel] to true if token associated
  // to a final state exists in the last token queue of that channel
  void GetBestCost(
      const std::vector<ChannelId> &channels, bool isfinal,
      std::vector<std::pair<int32, CostType>> *argmins,
      std::vector<std::vector<std::pair<int, float>>> *list_lattice_tokens,
      std::vector<bool> *has_reached_final);

  // Fills *out_nonempty_channels with channels with NumFramesDecoded(ichannel)
  // > 0
  void FillWithNonEmptyChannels(const std::vector<ChannelId> &channels,
                                std::vector<ChannelId> *out_nonempty_channels);

  // Given a token, get its best predecessor (lower cost predecessor)
  // Used by GetBestPath or best path traceback
  void GetBestPredecessor(int32 ichannel, int32 curr_token_idx,
                          int32 *prev_token_idx_out, int32 *arc_idx_out);
  // Expand the arcs, emitting stage. Must be called after
  // a preprocess_in_place, which happens in PostProcessingMainQueue.
  // ExpandArcsEmitting is called first when decoding a frame,
  // using the preprocessing that happened at the end of the previous
  // frame, in PostProcessingMainQueue
  void ExpandArcsEmitting();
  // ExpandArcs, non-emitting stage. Must be called after
  // PruneAndPreprocess.
  void ExpandArcsNonEmitting();
  // If we have more than max_active_ tokens in the queue (either after an
  // expand, or at the end of the frame)
  // we will compute a new beam that will only keep a number of tokens as
  // close as possible to max_active_ tokens (that number is >=
  // max_active_) (soft topk) All ApplyMaxActiveAndReduceBeam is find the
  // right beam for that topk and set it. We need to then call
  // PruneAndPreprocess (explicitly pruning tokens with cost > beam) Or
  // PostProcessingMainQueue (ignoring tokens with cost > beam in the next
  // frame)
  void ApplyMaxActiveAndReduceBeam(enum QUEUE_ID queue_id);
  // Called after an ExpandArcs. Prune the aux_q (output of the
  // ExpandArcs), move the survival tokens to the main_q, do the
  // preprocessing at the same time We don't need it after the last
  // ExpandArcsNonEmitting.
  void PruneAndPreprocess();
  // Once the non-emitting is done, the main_q is final for that frame.
  // We now generate all the data associated with that main_q, such as
  // listing the different tokens sharing the same token.next_state we
  // also preprocess for the ExpandArcsEmitting of the next frame Once
  // PostProcessingMainQueue, all working data is back to its original
  // state, to make sure we're ready for the next context switch
  void PostProcessingMainQueue();
  // Moving the relevant data to host, ie the data that will be needed in
  // GetBestPath/GetRawLattice.
  // Happens when PostProcessingMainQueue is done generating that data
  void CopyMainQueueDataToHost();
  // CheckOverflow
  // If a kernel sets the flag h_q_overflow, we send a warning to stderr
  // Overflows are detected and prevented on the device. It only means
  // that we've discarded the tokens that were created after the queue was
  // full That's why we only send a warning. It is not a fatal error
  void CheckOverflow();
  // Evaluates the function func for each lane, returning the max of all
  // return values (func returns int32) Used for instance to ge the max
  // number of arcs for all lanes func is called with
  // h_lanes_counters_[ilane] for each lane. h_lanes_counters_ must be
  // ready to be used when calling GetMaxForAllLanes (you might want to
  // call
  // CopyLaneCountersToHost[A|]sync to make sure everything is ready
  // first)
  int32 GetMaxForAllLanes(std::function<int32(const LaneCounters &)> func);
  // Copy the lane counters back to host, async or sync
  // The lanes counters contain all the information such as main_q_end
  // (number of tokens in the main_q) main_q_narcs (number of arcs) during
  // the computation. That's why we frequently copy it back to host to
  // know what to do next
  void CopyLaneCountersToHostAsync();
  void CopyLaneCountersToHostSync();
  // The selected tokens for each frame will be copied back to host. We
  // will store them on host memory, and we wil use them to create the
  // final lattice once we've reached the last frame We will also copy
  // information on those tokens that we've generated on the device, such
  // as which tokens are associated to the same FST state in the same
  // frame, or their extra cost. We cannot call individuals Device2Host
  // copies for each channel, because it would lead to a lot of small
  // copies, reducing performance. Instead we concatenate all channels
  // data into a single continuous array, copy that array to host, then
  // unpack it to the individual channel vectors The first step (pack then
  // copy to host, async) is done in ConcatenateData The second step is
  // done in LaunchD2H and sLaunchH2HCopies A sync on cudaStream st has to
  // happen between the two functions to make sure that the copy is done
  //
  // Each lane contains X elements to be copied, where X = func(ilane)
  // That data is contained in the array (pointer, X), with pointer =
  // src[ilane] It will be concatenated in d_concat on device, then copied
  // async into h_concat That copy is launched on stream st The offset of
  // the data of each lane in the concatenate array is saved in
  // *lanes_offsets_ptr
  // it will be used for unpacking in MoveConcatenatedCopyToVector
  //
  // func is called with h_lanes_counters_[ilane] for each lane.
  // h_lanes_counters_
  // must be ready to be used when calling GetMaxForAllLanes (you might
  // want to call CopyLaneCountersToHost[A|]sync to make sure everything
  // is ready first) Concatenate data on device before calling the D2H
  // copies
  void ConcatenateData();
  // Start the D2H copies used to send data back to host at the end of
  // each frames
  void LaunchD2HCopies();
  // ComputeH2HCopies
  // At the end of each frame, we copy data back to host
  // That data was concatenated into a single continous array
  // We then have to unpack it and move it inside host memory
  // This is done by ComputeH2HCopies
  void ComputeH2HCopies();

  // Used to generate the partial hypotheses
  // Called by the worker threads async
  void BuildPartialHypothesisOutput(
      ChannelId ichannel,
      std::stack<std::pair<int, PartialPathArc *>> *traceback_buffer_);
  void GeneratePartialPath(LaneId ilane, ChannelId ichannel);

  void EndpointDetected(LaneId ilane, ChannelId ichannel);
  // Wait for the async partial hypotheses related tasks to be done
  // before returning
  void WaitForPartialHypotheses();

  // Takes care of preparing the data for ComputeH2HCopies
  // and check whether we can use the threadpool or we have to do the work
  // on the current thread
  void LaunchH2HCopies();
  // Function called by the CPU worker threads
  // Calls ComputeH2HCopies when triggered
  void ComputeH2HCopiesCPUWorker();

  template <typename T>
  void MoveConcatenatedCopyToVector(const LaneId ilane,
                                    const ChannelId ichannel,
                                    const std::vector<int32> &lanes_offsets,
                                    T *h_concat,
                                    std::vector<std::vector<T>> *vecvec);
  void WaitForH2HCopies();
  void WaitForInitDecodingH2HCopies();
  // Computes a set of static asserts on the static values
  // In theory we should do them at compile time
  void CheckStaticAsserts();
  // Can be called in GetRawLattice to do a bunch of deep asserts on the
  // data Slow, so disabled by default
  void DebugValidateLattice();

  //
  // Data members
  //

  CudaDecoderConfig config_;
  const fst::SymbolTable *word_syms_;  // for partial hypotheses
  bool generate_partial_hypotheses_;   // set by AllowPartialHypotheses
  bool endpointing_;
  bool partial_traceback_;
  BaseFloat frame_shift_seconds_;

  std::set<int32> silence_phones_;

  // The CudaFst data structure contains the FST graph
  // in the CSR format, on both the GPU and CPU memory
  const CudaFst& fst_;

  // Counters used by a decoder lane
  // Contains all the single values generated during computation,
  // such as the current size of the main_q, the number of arcs currently
  // in that queue We load data from the channel state during
  // context-switch (for instance the size of the last token queue for
  // that channel)
  HostLaneMatrix<LaneCounters> h_lanes_counters_;
  // Counters of channels
  // Contains all the single values saved to remember the state of a
  // channel not used during computation. Those values are loaded/saved
  // into/from a lane during context switching
  ChannelCounters *h_channels_counters_;
  // Contain the various counters used by lanes/channels, such as
  // main_q_end, main_q_narcs. On device memory (equivalent of
  // h_channels_counters on device)
  DeviceChannelMatrix<ChannelCounters> d_channels_counters_;
  DeviceLaneMatrix<LaneCounters> d_lanes_counters_;
  // Number of lanes and channels, as defined in the constructor arguments
  int32 nlanes_, nchannels_;

  // We will now define the data used on the GPU
  // The data is mainly linked to two token queues
  // - the main queue
  // - the auxiliary queue
  //
  // The auxiliary queue is used to store the raw output of ExpandArcs.
  // We then prune that aux queue (and apply max-active) and move the
  // survival tokens in the main queue. Tokens stored in the main q can
  // then be used to generate new tokens (using ExpandArcs) We also
  // generate more information about what's in the main_q at the end of a
  // frame (in PostProcessingMainQueue)
  //
  // As a reminder, here's the data structure of a token :
  //
  // struct Token { state, cost, prev_token, arc_idx }
  //
  // Please keep in mind that this structure is also used in the context
  // of lattice decoding. We are not storing a list of forward links like
  // in the CPU decoder. A token stays an instanciation of an single arc.
  //
  // For performance reasons, we split the tokens in three parts :
  // { state } , { cost }, { prev_token, arc_idx }
  // Each part has its associated queue
  // For instance, d_main_q_state[i], d_main_q_cost[i], d_main_q_info[i]
  // all refer to the same token (at index i)
  // The data structure InfoToken contains { prev_token, arc_idx }
  // We also store the acoustic costs independently in
  // d_main_q_acoustic_cost_
  //
  // The data is eiher linked to a channel, or to a lane.
  //
  // Channel data (DeviceChannelMatrix):
  //
  // The data linked with a channel contains the data of frame i we need
  // to remember to compute frame i+1. It is the list of tokens from frame
  // i, with some additional info (ie the prefix sum of the emitting arcs
  // degrees from those tokens). We are only storing
  // d_main_q_state_and_cost_ as channel data because that's all we need
  // in a token to compute frame i+1. We don't need token.arc_idx or
  // token.prev_token. The reason why we also store that prefix sum is
  // because we do the emitting preprocessing at the end of frame i. The
  // reason for that is that we need infos from the hashmap to do that
  // preprocessing. The hashmap is always cleared at the end of a frame.
  // So we need to do the preprocessing at the end of frame i, and then
  // save d_main_q_degrees_prefix_sum_. d_main_q_arc_offsets is generated
  // also during preprocessing.
  //
  // Lane data (DeviceLaneMatrix):
  //
  // The lane data is everything we use during computation, but which we
  // reset at the end of each frame. For instance we use a hashmap at some
  // point during the computation, but at the end of each frame we reset
  // it. That way that hashmap is able to compute whichever channel the
  // next time AdvanceDecoding is called. The reasons why we do that is :
  //
  // - We use context switching. Before and after every frames, we can do
  // a context switching. Which means that a lane cannot save a channel's
  // state in any way once AdvanceDecoding returns. e.g., during a call of
  // AdvanceDecoding, ilane=2 may compute 5 frames from channel=57 (as
  // defined in the std::vector<ChannelId> channels). In the next call,
  // the same ilane=2 may compute 10 frames from channel=231. A lane data
  // has to be reset to its original state at the end of each
  // AdvanceDecoding call.
  // If somehow some data has to be saved, it needs to be declared as
  // channel data.
  //
  // - The reason why we make the distinction between lane and channel
  // data (in theory everything could be consider channel data), is
  // because a lane uses more memory than a channel. In the context of
  // online decoding, we need to create a lot channels, and we need them
  // to be as small as possible in memory. Everything that can be reused
  // between channels is stored as lane data.

  //
  // Channel data members:
  //

  DeviceChannelMatrix<int2> d_main_q_state_and_cost_;
  // Prefix sum of the arc's degrees in the main_q. Used by ExpandArcs,
  // set in the preprocess stages (either PruneAndPreprocess or
  // preprocess_in_place in PostProcessingMainQueue)
  DeviceChannelMatrix<int32> d_main_q_degrees_prefix_sum_;
  // d_main_q_arc_offsets[i] = fst_.arc_offsets[d_main_q_state[i]]
  // we pay the price for the random memory accesses of fst_.arc_offsets
  // in the preprocess kernel we cache the results in d_main_q_arc_offsets
  // which will be read in a coalesced fashion in expand
  DeviceChannelMatrix<int32> d_main_q_arc_offsets_;

  //
  // Lane data members:
  //

  // InfoToken
  // Usually contains {prev_token, arc_idx}
  // If more than one token is associated to a fst_state,
  // it will contain where to find the list of those tokens in
  // d_main_q_extra_prev_tokens
  // ie {offset,size} in that list. We differentiate the two situations by
  // calling InfoToken.IsUniqueTokenForStateAndFrame()
  DeviceLaneMatrix<InfoToken> d_main_q_info_;
  // Acoustic cost of a given token
  DeviceLaneMatrix<CostType> d_main_q_acoustic_cost_;
  // At the end of a frame, we use a hashmap to detect the tokens that are
  // associated with the same FST state S
  // We do it that the very end, to only use the hashmap on post-prune,
  // post-max active tokens
  DeviceLaneMatrix<HashmapValueT> d_hashmap_values_;
  // Reminder: in the GPU lattice decoder, a token is always associated
  // to a single arc. Which means that multiple tokens in the same frame
  // can be associated with the same FST state.
  //
  // We are NOT listing those duplicates as ForwardLinks in an unique
  // meta-token like in the CPU lattice decoder
  //
  // When more than one token is associated to a single FST state,
  // we will list those tokens into another list :
  // d_main_q_extra_prev_tokens we will also save data useful in such a
  // case, such as the extra_cost of a token compared to the best for that
  // state
  DeviceLaneMatrix<InfoToken> d_main_q_extra_prev_tokens_;
  DeviceLaneMatrix<float2> d_main_q_extra_and_acoustic_cost_;
  // Histogram. Used to perform the histogram of the token costs
  // in the main_q. Used to perform a soft topk of the main_q (max-active)
  DeviceLaneMatrix<int32> d_histograms_;
  // When filling the hashmap in PostProcessingMainQueue, we create a
  // hashmap value for each FST state presents in the main_q (if at least
  // one token is associated with that state)
  // d_main_q_state_hash_idx_[token_idx] is the index of the state
  // token.state in the hashmap Stored into a FSTStateHashIndex, which is
  // actually a int32. FSTStateHashIndex should only be accessed through
  // [Get|Set]FSTStateHashIndex, because it uses the bit sign to also
  // remember if that token is the representative of that state. If only
  // one token is associated with S, its representative will be itself
  DeviceLaneMatrix<FSTStateHashIndex> d_main_q_state_hash_idx_;
  // local_idx of the extra cost list for a state
  // For a given state S, first token associated with S will have
  // local_idx=0 the second one local_idx=1, etc. The order of the
  // local_idxs is random
  DeviceLaneMatrix<int32> d_main_q_n_extra_prev_tokens_local_idx_;
  // Where to write the extra_prev_tokens in the
  // d_main_q_extra_prev_tokens_ queue
  DeviceLaneMatrix<int32> d_main_q_extra_prev_tokens_prefix_sum_;
  // Used when computing the prefix_sums in preprocess_in_place. Stores
  // the local_sums per CTA
  DeviceLaneMatrix<int2> d_main_q_block_sums_prefix_sum_;
  // Defining the aux_q. Filled by ExpandArcs.
  // The tokens are moved to the main_q by PruneAndPreprocess
  DeviceLaneMatrix<int2> d_aux_q_state_and_cost_;
  DeviceLaneMatrix<InfoToken> d_aux_q_info_;
  // Dedicated space for the concat of extra_cost. We should reuse memory
  DeviceLaneMatrix<float2> d_extra_and_acoustic_cost_concat_matrix_;
  DeviceLaneMatrix<InfoToken> d_extra_prev_tokens_concat_matrix_;
  DeviceLaneMatrix<CostType> d_acoustic_cost_concat_matrix_;
  DeviceLaneMatrix<InfoToken> d_infotoken_concat_matrix_;
  // We will list in d_list_final_tokens_in_main_q all tokens within
  // [min_cost; min_cost+lattice_beam] It is used when calling GetBestCost
  // We only use an interface here because we will actually reuse data
  // from d_aux_q_state_and_cost We are done using the aux_q when
  // GetBestCost is called, so we can reuse that memory
  HostLaneMatrix<int2> h_list_final_tokens_in_main_q_;
  // Parameters used by the kernels
  // DeviceParams contains all the parameters that won't change
  // i.e. memory address of the main_q for instance
  // KernelParams contains information that can change.
  // For instance which channel is executing on which lane
  std::unique_ptr<DeviceParams> h_device_params_;
  std::unique_ptr<KernelParams> h_kernel_params_;
  std::vector<ChannelId> channel_to_compute_;
  int32 nlanes_used_;  // number of lanes used in h_kernel_params_
  // Initial lane
  // When starting a new utterance,
  // init_channel_id is used to initialize a channel
  int32 init_channel_id_;
  // CUDA streams used by the decoder
  cudaStream_t compute_st_, copy_st_;
  // Parameters extracted from CudaDecoderConfig
  // Those are defined in CudaDecoderConfig
  CostType default_beam_;
  CostType lattice_beam_;
  int32 ntokens_pre_allocated_;
  int32 max_active_;  // Target value from the parameters
  int32 aux_q_capacity_;
  int32 main_q_capacity_;
  // Hashmap capacity. Multiple of max_tokens_per_frame
  int32 hashmap_capacity_;
  // Static segment of the adaptive beam. Cf InitDeviceParams
  int32 adaptive_beam_static_segment_;
  // The first index of all the following vectors (or vector<vector>)
  // is the ChannelId. e.g., to get the number of frames decoded in
  // channel 2, look into num_frames_decoded_[2].

  // Keep track of the number of frames decoded in the current file.
  std::vector<int32> num_frames_decoded_;
  // Offsets of each frame in h_all_tokens_info_
  std::vector<std::vector<int32>> frame_offsets_;
  // Data storage. We store on host what we will need in
  // GetRawLattice/GetBestPath
  std::vector<std::vector<InfoToken>> h_all_tokens_info_;
  std::vector<std::vector<CostType>> h_all_tokens_acoustic_cost_;
  std::vector<std::vector<InfoToken>> h_all_tokens_extra_prev_tokens_;
  std::vector<std::vector<float2>>
      h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_;
  //TODO(hugovbraun): At some point we should switch to a shared_lock to be
  // able to compute partial lattices while still streaming new data for this
  // channel.
  std::vector<std::mutex> channel_lock_;

  // For each channel, set by PrepareForGetRawLattice
  // argmin cost, list of the tokens within
  // [best_cost;best_cost+lattice_beam] and if we've reached a final
  // token. Set by PrepareForGetRawLattice.
  std::vector<std::pair<int32, CostType>> h_all_argmin_cost_;
  std::vector<std::vector<std::pair<int, float>>> h_all_final_tokens_list_;
  std::vector<bool> h_all_has_reached_final_;
  // Buffer to store channels with nframes > 0.
  std::vector<ChannelId> nonempty_channels_;

  // Pinned memory arrays. Used for the DeviceToHost copies
  float2 *h_extra_and_acoustic_cost_concat_, *d_extra_and_acoustic_cost_concat_;
  InfoToken *h_infotoken_concat_, *d_infotoken_concat_;
  CostType *h_acoustic_cost_concat_, *d_acoustic_cost_concat_;
  InfoToken *h_extra_prev_tokens_concat_, *d_extra_prev_tokens_concat_;
  // second memory space used for double buffering
  float2 *h_extra_and_acoustic_cost_concat_tmp_;
  InfoToken *h_infotoken_concat_tmp_;
  CostType *h_acoustic_cost_concat_tmp_;
  InfoToken *h_extra_prev_tokens_concat_tmp_;
  // Offsets used in MoveConcatenatedCopyToVector
  std::vector<int32> h_main_q_end_lane_offsets_;
  std::vector<int32> h_emitting_main_q_end_lane_offsets_;
  std::vector<int32> h_n_extra_prev_tokens_lane_offsets_;
  // Index of the best index for the last frame. Used by endpointing/partial
  // results

  std::vector<BestPathTracebackHead> h_best_path_traceback_head_;
  std::vector<BestPathTracebackHead>
      h_all_channels_prev_best_path_traceback_head_;
  // Partial path so far on a given channel

  // Partial hypotheses to be used by user
  // Only valid between API calls (InitDecoding, AdvanceDecoding)
  std::vector<PartialHypothesis> h_all_channels_partial_hypotheses_out_;
  std::vector<char>
      h_all_channels_endpoint_detected_;  // not using a bool, we need it to be
                                          // threadsafe

  // Used internally to store the state of the current partial hypotheses
  std::vector<std::list<PartialPathArc>> h_all_channels_partial_hypotheses_;

  // Used when calling GetBestCost
  std::vector<std::pair<int32, CostType>> argmins_;
  std::vector<bool> has_reached_final_;
  std::vector<std::vector<std::pair<int32, CostType>>>
      list_finals_token_idx_and_cost_;
  bool compute_max_active_;
  cudaEvent_t nnet3_done_evt_;
  cudaEvent_t d2h_copy_acoustic_evt_;
  cudaEvent_t d2h_copy_infotoken_evt_;
  cudaEvent_t d2h_copy_extra_prev_tokens_evt_;
  cudaEvent_t concatenated_data_ready_evt_;
  cudaEvent_t lane_offsets_ready_evt_;
  // GetRawLattice helper
  // Data used when building the lattice in GetRawLattice

  // few typedef to make GetRawLattice easier to understand
  // Returns a unique id for each (iframe, fst_state) pair
  // We need to be able to quickly identity a (iframe, fst_state) ID
  //
  // A lattice state is defined by the pair (iframe, fst_state)
  // A token is associated to a lattice state (iframe, token.next_state)
  // Multiple token in the same frame can be associated to the same
  // lattice state (they all go to the same token.next_state) We need to
  // quickly identify what is the lattice state of a token. We are able to
  // do that through GetLatticeStateInternalId(token), which returns the
  // internal unique ID for each lattice state for a token
  //
  // When we build the output lattice, we a get new lattice state
  // output_lattice_state = fst_out->AddState()
  // We call this one OutputLatticeState
  // The conversion between the two is done through maps
  // [curr|prev]_f_raw_lattice_state_
  typedef int32 LatticeStateInternalId;
  typedef StateId OutputLatticeState;
  typedef int32 TokenId;
  LatticeStateInternalId GetLatticeStateInternalId(int32 total_ntokens,
                                                   TokenId token_idx,
                                                   InfoToken token);
  // Keeping track of a variety of info about states in the lattice
  // - token_extra_cost. A path going from the current lattice_state to
  // the end has an extra cost compared to the best path (which has an
  // extra cost of 0). token_extra_cost is the minimum of the extra_cost
  // of all paths going from the current lattice_state to the final frame.
  // - fst_lattice_state is the StateId of the lattice_state in fst_out
  // (in the output lattice). lattice_state is an internal state used in
  // GetRawLattice.
  // - is_state_closed is true if the token_extra_cost has been read by
  // another token. It means that the
  // token_extra_cost value has been used, and if we modify
  // token_extra_cost again, we may need to recompute the current frame
  // (so that everyone uses the latest token_extra_cost value)
  struct RawLatticeState {
    CostType token_extra_cost;
    OutputLatticeState fst_lattice_state;
    bool is_state_closed;
  };
  // extra_cost_min_delta_ used in the must_replay_frame situation. Please
  // read comments associated with must_replay_frame in GetRawLattice to
  // understand what it does
  CostType extra_cost_min_delta_;
  ThreadPoolLight *thread_pool_;
  std::vector<std::thread> cpu_dedicated_threads_;
  int32 n_threads_used_;
  std::vector<ChannelId> lanes2channels_todo_;
  std::atomic<int> n_acoustic_h2h_copies_todo_;
  std::atomic<int> n_extra_prev_tokens_h2h_copies_todo_;
  //TODO(hugovbraun): unused: std::atomic<int> n_d2h_copies_ready_;
  std::atomic<int> n_infotoken_h2h_copies_todo_;
  int32 n_h2h_task_not_done_;
  int32 n_init_decoding_h2h_task_not_done_;
  std::atomic<int> n_h2h_main_task_todo_;
  std::mutex n_h2h_task_not_done_mutex_;
  std::mutex n_init_decoding_h2h_task_not_done_mutex_;
  std::mutex n_h2h_main_task_todo_mutex_;
  std::condition_variable n_h2h_main_task_todo_cv_;
  std::condition_variable h2h_done_;
  std::condition_variable init_decoding_h2h_done_;
  //TODO(hugovbraun): unused: std::atomic<bool> active_wait_;

  // Used for sync on partial hypotheses tasks
  std::atomic<std::int32_t> n_partial_traceback_threads_todo_;
  std::atomic<std::int32_t> n_partial_traceback_threads_not_done_;

  // Set to false in destructor to stop threads.
  volatile bool h2h_threads_running_;

  // Using the output from GetBestPath, we add the best tokens (as
  // selected in GetBestCost) from the final frame to the output lattice.
  // We also fill the data structures (such as q_curr_frame_todo_, or
  // curr_f_raw_lattice_state_) accordingly
  void AddFinalTokensToLattice(
      ChannelId ichannel,
      std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      Lattice *fst_out);
  // Check if a token should be added to the lattice. If it should, then
  // keep_arc will be true
  void ConsiderTokenForLattice(
      ChannelId ichannel, int32 iprev, int32 total_ntokens, TokenId token_idx,
      OutputLatticeState fst_lattice_start, InfoToken *tok_beg,
      float2 *arc_extra_cost_beg, CostType token_extra_cost,
      TokenId list_prev_token_idx, int32 list_arc_idx,
      InfoToken *list_prev_token, CostType *this_arc_prev_token_extra_cost,
      CostType *acoustic_cost, OutputLatticeState *lattice_src_state,
      bool *keep_arc, bool *dbg_found_zero);
  // Add the arc to the lattice. Also updates what needs to be updated in
  // the GetRawLattice datastructures.
  void AddArcToLattice(
      int32 list_arc_idx, TokenId list_prev_token_idx,
      InfoToken list_prev_token, int32 curr_frame_offset,
      CostType acoustic_cost, CostType this_arc_prev_token_extra_cost,
      LatticeStateInternalId src_state_internal_id,
      OutputLatticeState fst_lattice_start,
      OutputLatticeState to_fst_lattice_state,
      std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
      std::vector<std::pair<TokenId, InfoToken>> *q_prev_frame_todo,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *prev_f_raw_lattice_state,
      std::unordered_set<int32> *f_arc_idx_added, Lattice *fst_out,
      bool *must_replay_frame);
  // Read a token information
  void GetTokenRawLatticeData(
      TokenId token_idx, InfoToken token, int32 total_ntokens,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      CostType *token_extra_cost, OutputLatticeState *to_fst_lattice_state);

  // A token is an instance of an arc. It goes to a FST state
  // (token.next_state) Multiple token in the same frame can go to the
  // same FST state. GetSameFSTStateTokenList returns that list
  void GetSameFSTStateTokenList(ChannelId ichannel, InfoToken &token,
                                InfoToken **tok_beg,
                                float2 **arc_extra_cost_beg, int32 *nprevs);

  // Swap datastructures at the end of a frame. prev becomes curr (we go
  // backward)
  //
  void SwapPrevAndCurrLatticeMap(
      int32 iframe, bool dbg_found_best_path,
      std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
      std::vector<std::pair<TokenId, InfoToken>> *q_prev_frame_todo,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *curr_f_raw_lattice_state,
      std::unordered_map<LatticeStateInternalId, RawLatticeState>
          *prev_f_raw_lattice_state,
      std::unordered_set<int32> *f_arc_idx_added);
};

}  // end namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA
#endif  // KALDI_CUDADECODER_CUDA_DECODER_H_
