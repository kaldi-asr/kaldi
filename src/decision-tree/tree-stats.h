// decision-tree/tree-stats.h

// Copyright 2015   Vimal Manohar

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

#include "util/stl-utils.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {
namespace decision_tree_classifier {

class BooleanTreeStats {
 public:
  void Write(std::ostream &os, bool binary) const;
  void WriteForMatlab(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary, bool add = false);
  int32 ReadAndWrite(std::istream &is, bool binary_in,
                    std::ostream &os, bool binary, 
                    bool write_for_matlab = false);

  void Accumulate(const std::vector<bool> &bits,
                  int32 label);
  void Accumulate(const std::pair<std::vector<bool>, 
                                  Vector<BaseFloat> > &stat);
  void Accumulate(const std::vector<bool> &vec, 
                                  const Vector<BaseFloat> &counts);

  void AddStats(const BooleanTreeStats &other);

  BooleanTreeStats(): num_classes_(-1) { };
  explicit BooleanTreeStats(int32 num_classes): num_classes_(num_classes) { };

  inline int32 NumClasses() const { return num_classes_; }
  inline int32 NumStats() const { return stats_.size(); };
  inline int32 NumBits() const { 
    return (NumStats() > 0 ? stats_[0].first.size() : -1);
  }; 

  const std::vector<std::pair<std::vector<bool>, Vector<BaseFloat> > >&  Stats() const { return stats_; }

 private:
  int32 num_classes_;
  std::vector<std::pair<std::vector<bool>, Vector<BaseFloat> > > stats_;
  std::unordered_map<std::vector<bool>, int32, std::hash<std::vector<bool> > > stats_map_;
};

}
}
