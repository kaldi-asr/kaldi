// -*- coding: utf-8 -*-
/* Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
 *               2012-2013  Vassil Panayotov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
 * MERCHANTABLITY OR NON-INFRINGEMENT.
 * See the Apache 2 License for the specific language governing permissions and
 * limitations under the License. */
#ifndef KALDI_DEC_WRAP_LATGEN_WRAPPER_H_
#define KALDI_DEC_WRAP_LATGEN_WRAPPER_H_
#include <string>
#include <vector>
#include "base/kaldi-types.h"

// forward declarations from Fst
namespace fst{
  template <typename Arc> class Fst;
  template <typename Weight> class ArcTpl; 
  template <class W> class TropicalWeightTpl;
  typedef TropicalWeightTpl<float> TropicalWeight;
  typedef ArcTpl<TropicalWeight> StdArc;
  typedef Fst<StdArc> StdFst;
  template<class FloatType> class LatticeWeightTpl;
  template <class A> class VectorFst;
  template<class WeightType, class IntType> class CompactLatticeWeightTpl; 

}
namespace kaldi{ 
  typedef fst::LatticeWeightTpl<BaseFloat> LatticeWeight;
  typedef fst::ArcTpl<LatticeWeight> LatticeArc;
  typedef fst::VectorFst<LatticeArc> Lattice;

  typedef fst::CompactLatticeWeightTpl<LatticeWeight, kaldi::int32> CompactLatticeWeight;
  typedef fst::ArcTpl<CompactLatticeWeight> CompactLatticeArc;
  typedef fst::VectorFst<CompactLatticeArc> CompactLattice;
}


namespace kaldi {

// forward declarations
class OnlBuffSource;
class Mfcc;
template<typename Mfcc> class OnlFeInput;
class OnlFeInput_Mfcc;
class OnlFeatInputItf;
class OnlFeatureMatrix;
class OnlDecodableDiagGmmScaled;
class TransitionModel;
class AmDiagGmm;
class OnlLatticeFasterDecoder;
class GmmLatgenWrapper;
struct OptionsItf;

struct KaldiDecoderGmmLatgenWrapperOptions  {
  /// Input sampling frequency is fixed to 16KHz
  explicit KaldiDecoderGmmLatgenWrapperOptions():kSampleFreq(16000), 
  acoustic_scale(0.1),
  lat_acoustic_scale(1.0), lat_lm_scale(1.0),
  left_context(4), right_context(4)
  {}
  int32 kSampleFreq;
  BaseFloat acoustic_scale;
  BaseFloat lat_acoustic_scale;
  BaseFloat lat_lm_scale;
  int32 left_context;
  int32 right_context;
  std::string model_rxfilename;
  std::string fst_rxfilename;
  std::string lda_mat_rspecifier;
  std::vector<int32> silence_phones;
  void Register(OptionsItf *po);
};


class GmmLatgenWrapper {
  public:
    GmmLatgenWrapper(): audio(NULL), mfcc(NULL), feat_input(NULL),
    feat_transform(NULL), feat_matrix(NULL), decodable(NULL),
    trans_model(NULL), amm(NULL), decoder(NULL), decode_fst(NULL) { }

    virtual ~GmmLatgenWrapper();
    size_t Decode(size_t max_frames);
    void FrameIn(unsigned char *frame, size_t frame_len);
    bool GetBestPath(std::vector<int> &v_out, BaseFloat *prob);
    bool GetRawLattice(fst::VectorFst<fst::StdArc> *fst_out);
    bool GetLattice(fst::VectorFst<fst::LogArc> * out_fst, double *tot_prob);
    void PruneFinal();
    void Reset(bool keep_buffer_data);
    bool Setup(int argc, char **argv);
  protected:
    OnlBuffSource *audio;
    Mfcc *mfcc;
    OnlFeInput<Mfcc> *feat_input;
    OnlFeatInputItf *feat_transform;
    OnlFeatureMatrix *feat_matrix;
    OnlDecodableDiagGmmScaled *decodable;
    TransitionModel *trans_model;
    AmDiagGmm *amm;
    OnlLatticeFasterDecoder *decoder;
    fst::StdFst *decode_fst;
  private:
    bool initialized_;
    BaseFloat lat_lm_scale_, lat_acoustic_scale_;
    void Deallocate();
};

} // namespace kaldi

#endif  // #ifdef KALDI_DEC_WRAP_LATGEN_WRAPPER_H_
