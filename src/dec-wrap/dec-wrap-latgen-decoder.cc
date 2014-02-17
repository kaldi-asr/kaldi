#include "dec-wrap/dec-wrap-latgen-decoder.h"
#include "decoder/lattice-faster-decoder.h"

namespace kaldi {


size_t OnlLatticeFasterDecoder::Decode(DecodableInterface *decodable, size_t max_frames) {
  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  size_t i = 0;
  for (;(!decodable->IsLastFrame(frame_-2)) && i < max_frames ; ++frame_, ++i) {
    active_toks_.resize(frame_+1); // new column

    ProcessEmitting(decodable, frame_);
      
    ProcessNonemitting(frame_);
    
    if(frame_ % config_.prune_interval == 0)
      PruneActiveTokens(frame_, config_.lattice_beam * 0.1); // use larger delta.        
  }
  // // Returns bigger than 0 if we have any kind of traceback available (not necessarily
  // // to the end state; query ReachedFinal() for that).
  // return final_costs_.size();
  return i;  // number of actually processed frames
}


void OnlLatticeFasterDecoder::Reset() {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  final_active_ = false;
  final_costs_.clear();
  num_toks_ = 0;
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL);
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;
  frame_ = 1;
}


}  // namespace kaldi
