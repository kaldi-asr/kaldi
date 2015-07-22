// nnet3/nnet-example-utils.cc

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

#include "nnet3/nnet-example-utils.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {


// get a sorted list of all feature names (will normally be just the string
// "input").
static void GetFeatureNames(const std::vector<NnetExample> &src,
                            std::vector<std::string> *names_vec) {
  std::set<std::string> names;
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<Feature>::const_iterator iter2 = iter->features.begin(),
                                          end2 = iter->features.end();
    for (; iter2 != end2; ++iter2)
      names.insert(iter2->name);
  }
  CopySetToVector(names, names_vec);
}

// get feature dimension for each input name (and make sure they are consistent;
// and get the feature "sizes" which are the total number of Indexes for that
// input (needed to correctly size the output matrix).
static void GetFeatureDimsAndSizes(const std::vector<NnetExample> &src,
                                   const std::vector<std::string> &names,
                                   std::vector<int32> *dims,
                                   std::vector<int32> *sizes) {
  dims->clear();
  dims->resize(names.size(), -1);
  sizes->clear();
  sizes->resize(names.size(), 0);
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<Feature>::const_iterator iter2 = iter->features.begin(),
                                          end2 = iter->features.end();
    for (; iter2 != end2; ++iter2) {
      const Feature &feat = *iter2;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, feat.name);
      KALDI_ASSERT(*names_iter == feat.name);
      int32 i = names_iter - names_begin;
      int32 this_dim = feat.features.NumCols();
      if ((*dims)[i] == -1)
        (*dims)[i] = this_dim;
      else if((*dims)[i] != this_dim) {
        KALDI_ERR << "Merging examples with inconsistent feature dims: "
                  << (*dims)[i] << " vs. " << this_dim << " for '"
                  << feat.name << "'.";
      }
      KALDI_ASSERT(feat.features.NumRows() == feat.indexes.size());      
      int32 this_size = feat.indexes.size();
      (*sizes)[i] += this_size;
    }
  }
}



// get a sorted list of all supervision names (will normally be just the string
// "output").
static void GetSupervisionNames(const std::vector<NnetExample> &src,
                                std::vector<std::string> *names_vec) {
  std::set<std::string> names;
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<Supervision>::const_iterator iter2 = iter->supervision.begin(),
                                              end2 = iter->supervision.end();
    for (; iter2 != end2; ++iter2)
      names.insert(iter2->name);
  }
  CopySetToVector(names, names_vec);
}



// get supervision (output) dimension for each output name (and make sure they
// are consistent; and get the corresponding "sizes" which are the total number
// of Indexes for that output.
static void GetSupervisionDimsAndSizes(const std::vector<NnetExample> &src,
                                       const std::vector<std::string> &names,
                                       std::vector<int32> *dims,
                                       std::vector<int32> *sizes) {
  dims->clear();
  dims->resize(names.size(), -1);
  sizes->clear();
  sizes->resize(names.size(), 0);
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<Supervision>::const_iterator iter2 = iter->supervision.begin(),
                                          end2 = iter->supervision.end();
    for (; iter2 != end2; ++iter2) {
      const Supervision &sup = *iter2;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, sup.name);
      KALDI_ASSERT(*names_iter == sup.name);
      int32 i = names_iter - names_begin;
      int32 this_dim = sup.dim;
      if ((*dims)[i] == -1)
        (*dims)[i] = this_dim;
      else if((*dims)[i] != this_dim) {
        KALDI_ERR << "Merging examples with inconsistent output dims: "
                  << (*dims)[i] << " vs. " << this_dim << " for '"
                  << sup.name << "'.";
      }
      KALDI_ASSERT(sup.labels.size() == sup.indexes.size());
      int32 this_size = sup.indexes.size();
      (*sizes)[i] += this_size;
    }
  }
}


// Do the final merging of features, once we have obtained the names, dims and
// sizes for each feature type.
static void MergeFeatures(const std::vector<NnetExample> &src,
                          const std::vector<std::string> &names,
                          const std::vector<int32> &dims,
                          const std::vector<int32> &sizes,
                          bool compress,
                          NnetExample *merged_eg) {
  int32 num_feats = names.size();
  std::vector<int32> cur_size(num_feats, 0);
  std::vector<Matrix<BaseFloat> > output_mats(num_feats);
  merged_eg->features.clear();
  merged_eg->features.resize(num_feats);
  for (int32 f = 0; f < num_feats; f++) {
    Feature &feat = merged_eg->features[f];
    int32 dim = dims[f], size = sizes[f];
    KALDI_ASSERT(dim > 0 && size > 0);
    feat.name = names[f];
    feat.indexes.resize(size);
    // feat.features is of type PossiblyCompressedMatrix; we'll set it up later,
    // but for now it's easier to deal with a regular matrix.
    output_mats[f].Resize(size, dim, kUndefined);
  }
  
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (int32 n = 0; iter != end; ++iter,++n) {
    std::vector<Feature>::const_iterator iter2 = iter->features.begin(),
                                          end2 = iter->features.end();
    for (; iter2 != end2; ++iter2) {
      const Feature &feat = *iter2;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, feat.name);
      KALDI_ASSERT(*names_iter == feat.name);
      int32 f = names_iter - names_begin;
      int32 this_dim = feat.features.NumCols(),
           this_size = feat.indexes.size(),
        &this_offset = cur_size[f];
      KALDI_ASSERT(this_dim == dims[f] &&
                   this_size + this_offset <= sizes[f]);
      Feature &output_feat = merged_eg->features[f];
      std::copy(feat.indexes.begin(), feat.indexes.end(),
                output_feat.indexes.begin() + this_offset);
      // Set the n index to be different for each of the original examples.
      for (int32 i = this_offset; i < this_offset + this_size; i++) {
        // we could easily support merging already-merged egs, but I don't see a
        // need for it right now.
        KALDI_ASSERT(output_feat.indexes[i].n == 0 &&
                     "Merging already-merged egs?");
        output_feat.indexes[i].n = n;
      }
      SubMatrix<BaseFloat> output_part(output_mats[f], this_offset, this_size,
                                       0, this_dim);
      feat.features.CopyToMat(&output_part);
      this_offset += this_size;  // note: this_offset is a reference.
    }
  }
  KALDI_ASSERT(cur_size == sizes);
  for (int32 f = 0; f < num_feats; f++) {
    merged_eg->features[f].features.Set(output_mats[f],
                                        compress);
    output_mats[f].Resize(0, 0);
  }
}


// Do the final merging of supervision information, once we have obtained the
// names, dims and sizes for each feature type.
static void MergeSupervision(const std::vector<NnetExample> &src,
                             const std::vector<std::string> &names,
                             const std::vector<int32> &dims,
                             const std::vector<int32> &sizes,
                             NnetExample *merged_eg) {
  int32 num_sup = names.size();
  std::vector<int32> cur_size(num_sup, 0);
  merged_eg->supervision.clear();
  merged_eg->supervision.resize(num_sup);
  for (int32 s = 0; s < num_sup; s++) {
    Supervision &sup = merged_eg->supervision[s];
    int32 dim = dims[s], size = sizes[s];
    KALDI_ASSERT(dim > 0 && size > 0);
    sup.name = names[s];
    sup.dim = dim;
    sup.indexes.resize(size);
    sup.labels.resize(size);
  }
  
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (int32 n = 0; iter != end; ++iter,++n) {
    std::vector<Supervision>::const_iterator iter2 = iter->supervision.begin(),
                                             end2 = iter->supervision.end();
    for (; iter2 != end2; ++iter2) {
      const Supervision &sup = *iter2;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, sup.name);
      KALDI_ASSERT(*names_iter == sup.name);
      int32 s = names_iter - names_begin;
      int32 this_dim = sup.dim,
           this_size = sup.indexes.size(),
        &this_offset = cur_size[s];
      KALDI_ASSERT(this_dim == dims[s] &&
                   this_size + this_offset <= sizes[s] &&
                   sup.indexes.size() == sup.labels.size());
      Supervision &output_sup = merged_eg->supervision[s];
      std::copy(sup.indexes.begin(), sup.indexes.end(),
                output_sup.indexes.begin() + this_offset);
      // Set the n index to be different for each of the original examples.
      for (int32 i = this_offset; i < this_offset + this_size; i++) {
        // we could easily support merging already-merged egs, but I don't see a
        // need for it right now.
        KALDI_ASSERT(output_sup.indexes[i].n == 0 &&
                     "Merging already-merged egs?");
        output_sup.indexes[i].n = n;
      }
      std::copy(sup.labels.begin(), sup.labels.end(),
                output_sup.labels.begin() + this_offset);
      this_offset += this_size;  // note: this_offset is a reference.
    }
  }
  KALDI_ASSERT(cur_size == sizes);
}



void MergeExamples(const std::vector<NnetExample> &src,
                   bool compress,
                   NnetExample *merged_eg) {
  int32 size = src.size();
  KALDI_ASSERT(size > 0);

  std::vector<std::string> feature_names,
      supervision_names;
  GetFeatureNames(src, &feature_names);
  GetSupervisionNames(src, &supervision_names);

  // the _sizes variables are the total number of Indexes
  // we have across all examples, for that input or output.
  std::vector<int32> feature_dims, feature_sizes,
      supervision_dims, supervision_sizes;
  GetFeatureDimsAndSizes(src, feature_names,
                         &feature_dims, &feature_sizes);
  GetSupervisionDimsAndSizes(src, supervision_names,
                             &supervision_dims, &supervision_sizes);
  MergeFeatures(src, feature_names, feature_dims, feature_sizes,
                compress, merged_eg);
  MergeSupervision(src, supervision_names, supervision_dims, supervision_sizes,
                   merged_eg);
}



} // namespace nnet3
} // namespace kaldi
