#include "stdio.h"
#include "iostream"
#include "stdlib.h"

typedef int int32;
typedef long long int64;
typedef float BaseFloat;

struct GPUArc{ 
    typedef int StateId;

    StateId curstate;
    Arc arc;
};


/* there are couple changes here:
 * 1. Elem* --> vector<Elem>& (Linked List --> Vector)
 * 2. list_head --> list
 */ 
__global__ BaseFloat GetCutOff(vector<Elem>& lists, size_t *tok_count,
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

__global__ BaseFloat ProcessEmitting(OnlineDecodableDiagGmmScaled* decodable){
  
}

__global__ void ProcessNonemitting(BaseFloat cutoff);

//ini gimana cara bikin supaya DecodableInterface diassign di CUDA
//EDIT : Pake kelasnya langsung aja, jadi pake OnlineDecodableDiagGmmScaled langsung
__global__ DecodeState Decode(OnlineDecodableDiagGmmScaled* decodable);
