// nnet3/nnet-example.cc

// Copyright 2012-2015    Johns Hopkins University (author: Daniel Povey)
//                2014    Vimal Manohar

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

#include "nnet3/nnet-example.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {

void Feature::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Feat>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  features.Write(os, binary);
  WriteToken(os, binary, "</Feat>");
  KALDI_ASSERT(static_cast<size_t>(features.NumRows()) == indexes.size());
}

void Feature::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Feat>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  features.Read(is, binary);
  ExpectToken(is, binary, "</Feat>");    
}

Feature::Feature(const std::string &name,
                 int32 t_begin, const MatrixBase<BaseFloat> &feats):
    name(name), features(feats) {
  int32 num_rows = feats.NumRows();
  KALDI_ASSERT(num_rows > 0);
  indexes.resize(num_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_rows; i++)
    indexes[i].t = t_begin + i;
}

Supervision::Supervision(const std::string &name,
                         int32 dim,
                         int32 t_begin,
                         const Posterior &labels):
    name(name), dim(dim), labels(labels) {
  int32 num_frames = labels.size();
  KALDI_ASSERT(num_frames > 0);
  indexes.resize(num_frames);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_frames; i++)
    indexes[i].t = t_begin + i;
  // do a spot-check of one of the label indexes, that it's less than dim.
  KALDI_ASSERT(!labels.empty() &&
               (labels.back().empty() || labels.back().back().first < dim));
  
}

void Supervision::Write(std::ostream &os, bool binary) const{
  WriteToken(os, binary, "<Sup>");
  WriteToken(os, binary, name);
  WriteBasicType(os, binary, dim);
  WriteIndexVector(os, binary, indexes);
  WritePosterior(os, binary, labels);
  WriteToken(os, binary, "</Sup>");
  KALDI_ASSERT(labels.size() == indexes.size());
}

void Supervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Sup>");
  ReadToken(is, binary, &name);
  ReadBasicType(is, binary, &dim);
  ReadIndexVector(is, binary, &indexes);
  ReadPosterior(is, binary, &labels);
  ExpectToken(is, binary, "</Sup>");
}

void NnetExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3Eg>");

  WriteToken(os, binary, "<Features>");
  int32 size = features.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    features[i].Write(os, binary);
  WriteToken(os, binary, "<Supervision>");  
  size = supervision.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Write(os, binary);
  WriteToken(os, binary, "</Nnet3Eg>");
}

void NnetExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3Eg>");
  ExpectToken(is, binary, "<Features>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size < 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  features.resize(size);
  for (int32 i = 0; i < size; i++)
    features[i].Read(is, binary);
  ExpectToken(is, binary, "<Supervision>");  
  ReadBasicType(is, binary, &size);
  if (size < 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  supervision.resize(size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3Eg>");
}


} // namespace nnet3
} // namespace kaldi
