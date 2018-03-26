#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


/* includes from other files */
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc

#include "hmm/transition-model.h"

using namespace kaldi;

/****** START OF OPTS ******/
struct FasterDecoderOptions {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  FasterDecoderOptions(): beam(16.0),
                          max_active(std::numeric_limits<int32>::max()),
                          min_active(20), // This decoder mostly used for
                                          // alignment, use small default.
                          beam_delta(0.5),
                          hash_ratio(2.0) { }
  void Register(OptionsItf *opts, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active,
                   "Decoder min active states (don't prune if #active less than this).");
    if (full) {
      opts->Register("beam-delta", &beam_delta,
                     "Increment used in decoder [obscure setting]");
      opts->Register("hash-ratio", &hash_ratio,
                     "Setting used in decoder to control hash behavior");
    }
  }
};

struct OnlineFasterDecoderOpts : public FasterDecoderOptions {
  BaseFloat rt_min; // minimum decoding runtime factor
  BaseFloat rt_max; // maximum decoding runtime factor
  int32 batch_size; // number of features decoded in one go
  int32 inter_utt_sil; // minimum silence (#frames) to trigger end of utterance
  int32 max_utt_len_; // if utt. is longer, we accept shorter silence as utt. separators
  int32 update_interval; // beam update period in # of frames
  BaseFloat beam_update; // rate of adjustment of the beam
  BaseFloat max_beam_update; // maximum rate of beam adjustment

  OnlineFasterDecoderOpts() :
    rt_min(.7), rt_max(.75), batch_size(27),
    inter_utt_sil(50), max_utt_len_(1500),
    update_interval(3), beam_update(.01),
    max_beam_update(0.05) {}

  void Register(OptionsItf *opts, bool full) {
    FasterDecoderOptions::Register(opts, full);
    opts->Register("rt-min", &rt_min,
                   "Approximate minimum decoding run time factor");
    opts->Register("rt-max", &rt_max,
                   "Approximate maximum decoding run time factor");
    opts->Register("update-interval", &update_interval,
                   "Beam update interval in frames");
    opts->Register("beam-update", &beam_update, "Beam update rate");
    opts->Register("max-beam-update", &max_beam_update, "Max beam update rate");
    opts->Register("inter-utt-sil", &inter_utt_sil,
                   "Maximum # of silence frames to trigger new utterance");
    opts->Register("max-utt-length", &max_utt_len_,
                   "If the utterance becomes longer than this number of frames, "
                   "shorter silence is acceptable as an utterance separator");
  }
};

/****** END OF OPTS ******/

/****** START OF NEW STRUCTS ******/
struct GPUArc{ 
    typedef int StateId;
    StateId curstate;
    Arc arc;
};

/* Declaration of Kernels*/

__global__ BaseFloat GPUGetCutOff(std::vector<Elem>&lists, size_t *tok_count,
                                  BaseFloat *adaptive_bean, Elem **best_elem);

class OnlineFasterDecoderGPU{
  public:

    enum DecodeState {
      kEndFeats = 1, // No more scores are available from the Decodable
      kEndUtt = 2, // End of utterance, caused by e.g. a sufficiently long silence
      kEndBatch = 4 // End of batch - end of utterance not reached yet
    };

    // functions
    // variables
    const OnlineFasterDecoderOpts opts_;
    const ConstIntegerSet<int32> silence_set_; // silence phones IDs
    const TransitionModel &trans_model_; // needed for trans-id -> phone conversion
    const BaseFloat max_beam_; // the maximum allowed beam
    BaseFloat &effective_beam_; // the currently used beam
    DecodeState state_; // the current state of the decoder
    int32 frame_; // the next frame to be processed
    int32 utt_frames_; // # frames processed from the current utterance
    Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
    Token *prev_immortal_tok_; // ... all currently active tokens

    class Token {
      public:
        Arc arc_; // contains only the graph part of the cost;
        // we can work out the acoustic part from difference between
        // "cost_" and prev->cost_.
        Token *prev_;
        int32 ref_count_;
        // if you are looking for weight_ here, it was removed and now we just have
        // cost_, which corresponds to ConvertToCost(weight_).
        double cost_;
        inline Token(const Arc &arc, BaseFloat ac_cost, Token *prev):
            arc_(arc), prev_(prev), ref_count_(1) {
          if (prev) {
            prev->ref_count_++;
            cost_ = prev->cost_ + arc.weight.Value() + ac_cost;
          } else {
            cost_ = arc.weight.Value() + ac_cost;
          }
        }
        inline Token(const Arc &arc, Token *prev):
            arc_(arc), prev_(prev), ref_count_(1) {
          if (prev) {
            prev->ref_count_++;
            cost_ = prev->cost_ + arc.weight.Value();
          } else {
            cost_ = arc.weight.Value();
          }
        }
        inline bool operator < (const Token &other) {
          return cost_ > other.cost_;
        }

      inline static void TokenDelete(Token *tok) {
        while (--tok->ref_count_ == 0) {
          Token *prev = tok->prev_;
          delete tok;
          if (prev == NULL) return;
          else tok = prev;
        }
      }
    };
};

/* there are couple changes here:
 * 1. Elem* --> vector<Elem>& (Linked List --> Vector) (((EDIT : Elem harus diganti sama sesuatu yang nampung StateId dan Token*
 * 2. list_head --> list
 */ 
__global__ BaseFloat GPUGetCutOff(std::vector<Elem>& lists, size_t *tok_count,
                                BaseFloat *adaptive_beam, Elem **best_elem){
  double best_cost = std::numeric_limits<double>::infinity();
  size_t count = 0;

  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {

    for (const Elem& e : lists){
      double w = e->val->cost_;
      if(w < best_cost){
        best_cost = w;
        if(best_elem) *best_elem = e;
      }
      count++:
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_cost + config_.beam;
  } else {
    tmp_array_.clear();
    for (const Elem& e : lists){
      double w = e->val->cost_;
      tmp_array_.push_back(w);
      if(w < best_cost){
        best_cost = w;
        if(best_elem) *best_elem = e;
      }
      count++:
    }
    if (tok_count != NULL) *tok_count = count;
    double beam_cutoff = best_cost + config_.beam,
        min_active_cutoff = std::numeric_limits<double>::infinity(),
        max_active_cutoff = std::numeric_limits<double>::infinity();
    
    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta;
      return max_active_cutoff;
    }    
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_cost;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

BaseFloat ParallelProcessEmitting(OnlineDecodableDiagGmmScaled* decodable){
  
}

void ParallelProcessNonemitting(BaseFloat cutoff);

//ini gimana cara bikin supaya DecodableInterface diassign di CUDA
//EDIT : Pake kelasnya langsung aja, jadi pake OnlineDecodableDiagGmmScaled langsung

DecodeState Decode(OnlineDecodableDiagGmmScaled* decodable){
  if (state_ == kEndFeats || state_ == kEndUtt) // new utterance
    ResetDecoder(state_ == kEndFeats);
  ParallelProcessNonemitting(std::numeric_limits<float>::max());
  int32 batch_frame = 0;
  Timer timer;
  double64 tstart = timer.Elapsed(), tstart_batch = tstart;
  BaseFloat factor = -1;
  for (; !decodable->IsLastFrame(frame_ - 1) && batch_frame < opts_.batch_size;
       ++frame_, ++utt_frames_, ++batch_frame) {
    if (batch_frame != 0 && (batch_frame % opts_.update_interval) == 0) {
      // adjust the beam if needed
      BaseFloat tend = timer.Elapsed();
      BaseFloat elapsed = (tend - tstart) * 1000;
      // warning: hardcoded 10ms frames assumption!
      factor = elapsed / (opts_.rt_max * opts_.update_interval * 10);
      BaseFloat min_factor = (opts_.rt_min / opts_.rt_max);
      if (factor > 1 || factor < min_factor) {
        BaseFloat update_factor = (factor > 1)?
            -std::min(opts_.beam_update * factor, opts_.max_beam_update):
             std::min(opts_.beam_update / factor, opts_.max_beam_update);
        effective_beam_ += effective_beam_ * update_factor;
        effective_beam_ = std::min(effective_beam_, max_beam_);
      }
      tstart = tend;
    }
    if (batch_frame != 0 && (frame_ % 200) == 0)
      // one log message at every 2 seconds assuming 10ms frames
      KALDI_VLOG(3) << "Beam: " << effective_beam_
          << "; Speed: "
          << ((timer.Elapsed() - tstart_batch) * 1000) / (batch_frame*10)
          << " xRT";
    BaseFloat weight_cutoff = ParallelProcessEmitting(decodable);
    ParallelProcessNonemitting(weight_cutoff);
  }
  if (batch_frame == opts_.batch_size && !decodable->IsLastFrame(frame_ - 1)) {
    if (EndOfUtterance())
      state_ = kEndUtt;
    else
      state_ = kEndBatch;
  } else {
    state_ = kEndFeats;
  }
  return state_;
}
