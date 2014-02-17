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

#include <fst/arc-map.h>
#include <fst/arc.h>
#include "fstext/fstext-lib.h"
#include "fstext/fstext-utils.h"
#include "feat/feature-mfcc.h"
#include "dec-wrap/dec-wrap-utils.h"
#include "dec-wrap/dec-wrap-audio-source.h"
#include "dec-wrap/dec-wrap-latgen-wrapper.h"
#include "dec-wrap/dec-wrap-feat-input.h"
#include "dec-wrap/dec-wrap-decodable.h"
#include "dec-wrap/dec-wrap-latgen-wrapper.h"
#include "dec-wrap/dec-wrap-latgen-decoder.h"
// debug
#include <fstream>
#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>


namespace kaldi {

void KaldiDecoderGmmLatgenWrapperOptions::Register(OptionsItf *po) {
  po->Register("left-context", &left_context, "Number of frames of left context");
  po->Register("right-context", &right_context, "Number of frames of right context");
  po->Register("acoustic-scale", &acoustic_scale,
              "Scaling factor for acoustic likelihoods for forward decoding");
  po->Register("lat-acoustic-scale", &lat_acoustic_scale,
              "Scaling factor for acoustic likelihoods after forward decoding");
  po->Register("lat-lm-scale", &lat_lm_scale,
              "Scaling factor for LM likelihoods after forward decoding");
}

void GmmLatgenWrapper::Deallocate() {
  initialized_ = false;
  delete audio; audio = NULL;
  delete mfcc; mfcc = NULL;
  delete feat_input; feat_input = NULL;
  delete feat_transform; feat_transform = NULL;
  delete feat_matrix; feat_matrix = NULL;
  delete decodable; decodable = NULL;
  delete trans_model; trans_model = NULL;
  delete amm; amm = NULL;
  delete decoder; decoder = NULL;
  delete decode_fst; decode_fst = NULL;
}

GmmLatgenWrapper::~GmmLatgenWrapper() {
  Deallocate();
}

size_t GmmLatgenWrapper::Decode(size_t max_frames) {
  if (! initialized_)
    return 0;
  return decoder->Decode(decodable, max_frames);
}


void GmmLatgenWrapper::FrameIn(unsigned char *frame, size_t frame_len) {
  if (! initialized_)
    return;
  audio->Write(frame, frame_len);
}


bool GmmLatgenWrapper::GetBestPath(std::vector<int> &out_ids, BaseFloat *prob) {
  *prob = -1.0;  // default value for failures
  if (! initialized_)
    return false;
  Lattice lat;
  bool ok = decoder->GetBestPath(&lat);
  LatticeWeight weight;
  fst::GetLinearSymbolSequence(lat,
                               static_cast<vector<int32> *>(0),
                               &out_ids,
                               &weight);

    *prob = weight.Value1() + weight.Value2();
  return ok;
}


bool GmmLatgenWrapper::GetLattice(fst::VectorFst<fst::LogArc> *fst_out, 
                                  double *tot_prob) {
  if (! initialized_)
    return false;

  CompactLattice lat;
  bool ok = decoder->GetLattice(&lat);

  KALDI_ASSERT(lat_acoustic_scale_ != 0.0 || lat_lm_scale_ != 0.0);
  BaseFloat ac_scale = 1.0, acoustic_scale = decodable->GetAcousticScale();
  if (acoustic_scale != 0.0) // We'll unapply the acoustic scaling for forward decoding
    ac_scale = 1.0 / acoustic_scale;
  // We apply new acoustic scaling (and we supposed the lattice is not scaled)
  ac_scale = ac_scale * lat_acoustic_scale_;
  fst::ScaleLattice(fst::LatticeScale(lat_lm_scale_, ac_scale), &lat);

  *tot_prob = CompactLatticeToWordsPost(lat, fst_out);  // TODO tot_prob is sensible?

  return ok;
}


bool GmmLatgenWrapper::GetRawLattice(fst::VectorFst<fst::StdArc> *fst_out) {
  // TODO probably to low level for the API
  if (! initialized_)
    return false;

  Lattice lat;
  bool ok = decoder->GetRawLattice(&lat);
  fst::Connect(&lat); // Will get rid of this later... shouldn't have any

  KALDI_ASSERT(lat_acoustic_scale_ != 0.0 || lat_lm_scale_ != 0.0);
  fst::ScaleLattice(fst::LatticeScale(lat_lm_scale_, lat_acoustic_scale_), &lat);
  ConvertLattice(lat, fst_out); // adds up the (lm,acoustic) costs

  return ok;
}


void GmmLatgenWrapper::PruneFinal() {
  if (! initialized_)
    return;
  decoder->PruneFinal();
}


void GmmLatgenWrapper::Reset(bool keep_buffer_data) {
  if (! initialized_)
    return;
  if (!keep_buffer_data) {
    audio->Reset();
    feat_input->Reset();
    feat_transform->Reset();
  }
  feat_matrix->Reset();
  decodable->Reset();
  decoder->Reset();
}


bool GmmLatgenWrapper::Setup(int argc, char **argv) {
  initialized_ = false;
  try {
    KaldiDecoderGmmLatgenWrapperOptions wrapper_opts;
    OnlFeatureMatrixOptions feature_reading_opts;
    MfccOptions mfcc_opts;
    LatticeFasterDecoderConfig decoder_opts;
    DeltaFeaturesOptions delta_feat_opts;
    OnlBuffSourceOptions au_opts;

    // Parsing options
    ParseOptions po("Utterance segmentation is done on-the-fly.\n"
      "The delta/delta-delta(2-nd order) features are produced.\n\n"
      "Usage: decoder-binary-name [options] <model-in>"
      "<fst-in> <silence-phones> \n\n"
      "Example: decoder-binary-name --max-active=4000 --beam=12.0 "
      "--acoustic-scale=0.0769 model HCLG.fst words.txt '1:2:3:4:5'");

    wrapper_opts.Register(&po);
    mfcc_opts.Register(&po);
    decoder_opts.Register(&po);
    feature_reading_opts.Register(&po);
    delta_feat_opts.Register(&po);
    au_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      // throw std::invalid_argument("Specify 3 or 4 arguments. See the usage in stderr");
      return false;
    }
    if (po.NumArgs() == 3)
      if (wrapper_opts.left_context % delta_feat_opts.order != 0 ||
          wrapper_opts.left_context != wrapper_opts.right_context)
        KALDI_ERR << "Invalid left/right context parameters!";

    wrapper_opts.model_rxfilename = po.GetArg(1);
    wrapper_opts.fst_rxfilename = po.GetArg(2);
    wrapper_opts.silence_phones = phones_to_vector(po.GetArg(3));
    wrapper_opts.lda_mat_rspecifier = po.GetOptArg(4);

    lat_lm_scale_ = wrapper_opts.lat_lm_scale;
    lat_acoustic_scale_ = wrapper_opts.lat_acoustic_scale;

    // Setting up components
    trans_model = new TransitionModel();
    amm = new AmDiagGmm();
    {
      bool binary;
      Input ki(wrapper_opts.model_rxfilename, &binary);
      trans_model->Read(ki.Stream(), binary);
      amm->Read(ki.Stream(), binary);
    }

    decode_fst = ReadDecodeGraph(wrapper_opts.fst_rxfilename);
    decoder = new OnlLatticeFasterDecoder(
                                    *decode_fst, decoder_opts);

    audio = new OnlBuffSource(au_opts);

    mfcc = new Mfcc(mfcc_opts);
    int32 frame_length = mfcc_opts.frame_opts.frame_length_ms;
    int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms;
    feat_input = new OnlFeInput<Mfcc>(audio, mfcc,
                               frame_length * (wrapper_opts.kSampleFreq / 1000),
                               frame_shift * (wrapper_opts.kSampleFreq / 1000));

    if (wrapper_opts.lda_mat_rspecifier != "") {
      bool binary_in;
      Matrix<BaseFloat> lda_transform;
      Input ki(wrapper_opts.lda_mat_rspecifier, &binary_in);
      lda_transform.Read(ki.Stream(), binary_in);
      // lda_transform is copied to OnlLdaInput
      feat_transform = new OnlLdaInput(feat_input,
                                lda_transform,
                                wrapper_opts.left_context, wrapper_opts.right_context);
      KALDI_VLOG(1) << "LDA will be used for decoding" << std::endl;
    } else {
      feat_transform = new OnlDeltaInput(delta_feat_opts, feat_input);
      KALDI_VLOG(1) << "Delta + delta-delta will be used for decoding" << std::endl;
    }

    feat_matrix = new OnlFeatureMatrix(feature_reading_opts,
                                       feat_transform);
    decodable = new OnlDecodableDiagGmmScaled(*amm,
                                            *trans_model,
                                            wrapper_opts.acoustic_scale, feat_matrix);

  } catch(const std::exception& e) {
    Deallocate();
    // throw e;
    std::cerr << e.what() << std::endl;
    return false;
  }
  initialized_ = true;
  return initialized_;
}

} // namespace kaldi
