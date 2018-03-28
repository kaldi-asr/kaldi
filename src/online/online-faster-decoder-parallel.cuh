#ifndef KALDI_ONLINE_ONLINE_FASTER_DECODER_CUH__
#define KALDI_ONLINE_ONLINE_FASTER_DECODER_CUH__

/* Keterangan Include:
 * itf/options-itf.h      : kelas OptionsItf
 * itf/decodable-itf.h    : kelas DecodableInterface
 */

#include "itf/options-itf.h"
#include "itf/decodable-itf.h"

namespace kaldi{

/****** START OF NEW STRUCTS ******/
struct GPUArc{ 
    typedef int StateId;
    StateId curstate;
    Arc arc;
};

struct OnlineFasterDecoderParallelOpts {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;

  BaseFloat rt_min; // minimum decoding runtime factor
  BaseFloat rt_max; // maximum decoding runtime factor
  int32 batch_size; // number of features decoded in one go
  int32 inter_utt_sil; // minimum silence (#frames) to trigger end of utterance
  int32 max_utt_len_; // if utt. is longer, we accept shorter silence as utt. separators
  int32 update_interval; // beam update period in # of frames
  BaseFloat beam_update; // rate of adjustment of the beam
  BaseFloat max_beam_update; // maximum rate of beam adjustment

  OnlineFasterDecoderParallelOptions(): beam(16.0),
    max_active(std::numeric_limits<int32>::max()),
    min_active(20), // This decoder mostly used for
                    // alignment, use small default.
    beam_delta(0.5),
    hash_ratio(2.0), 
    rt_min(.7), rt_max(.75), batch_size(27),
    inter_utt_sil(50), max_utt_len_(1500),
    update_interval(3), beam_update(.01),
    max_beam_update(0.05) {}

  void Register(OptionsItf *opts, bool full) {
    
    // just copied from FasterDecoderOptions
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

    // just copied from OnlineFasterDecoderOpts
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

/****** END OF NEW STRUCTS ******/

/* Declaration of Kernels*/
// NOTE : The parameters will be changed because of OOP reasons.

/* Class OnlineFasterDecoderParallel */
// This class is combined from online/online-faster-decoder.h and decoder/faster-decoder.h.

class OnlineFasterDecoderParallel {

  public:
    typedef fst::StdArc Arc;
    typedef Arc::Label Label;
    typedef Arc::StateId StateId;
    typedef Arc::Weight Weight;
    enum DecodeState {
      kEndFeats = 1, // No more scores are available from the Decodable
      kEndUtt = 2, // End of utterance, caused by e.g. a sufficiently long silence
      kEndBatch = 4 // End of batch - end of utterance not reached yet
    };


    /* Changes here :
     * effective_beam_ is assigned same with max_beam
     */ 
    OnlineFasterDecoderParallel(
      const fst::Fst<fst::StdArc> &fst,
      const FasterDecoderOptions &opts,
      const std::vector<int32> &sil_phones,
      const TransitionModel &trans_model) : 
        fst_(fst), opts_(opts), num_frames_decoded(-1),
        silence_set_(sil_phones), trans_model_(trans_model),
        max_beam_(opts.beam), effective_beam_(opts.beam),
        state_(kEndFeats), frame_(0), utt_frames_(0) {}

    ~OnlineFasterDecoderParallel(); // TODO : ini nanti liat yang faster-decoder.h

    /// As a new alternative to Decode(), you can call InitDecoding
    /// and then (possibly multiple times) AdvanceDecoding().
    void InitDecoding();
    DecodeState Decode(DecodableInterface *decodable);

    // Makes a linear graph, by tracing back from the last "immortal" token
    // to the previous one
    bool PartialTraceback(fst::MutableFst<LatticeArc> *out_fst);

    // Makes a linear graph, by tracing back from the best currently active token
    // to the last immortal token. This method is meant to be invoked at the end
    // of an utterance in order to get the last chunk of the hypothesis
    void FinishTraceBack(fst::MutableFst<LatticeArc> *fst_out);

    // Returns "true" if the best current hypothesis ends with long enough silence
    bool EndOfUtterance();

  protected:
    const OnlineFasterDecoderParallelOpts opts_;
    const ConstIntegerSet<int32> silence_set_; // silence phones IDs
    const TransitionModel &trans_model_; // needed for trans-id -> phone conversion
    const BaseFloat max_beam_; // the maximum allowed beam
    BaseFloat &effective_beam_; // the currently used beam
    DecodeState state_; // the current state of the decoder
    int32 frame_; // the next frame to be processed
    int32 utt_frames_; // # frames processed from the current utterance
    Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
    Token *prev_immortal_tok_; // ... all currently active tokens

    const fst::Fst<fst::StdArc> &fst_;
    std::vector<StateId> queue_;  // temp variable used in ProcessNonemitting,
    std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
    // make it class member to avoid internal new/delete.

    // Keep track of the number of frames decoded in the current file.
    int32 num_frames_decoded_;

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

    KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineFasterDecoderParallel);
};

}

#endif