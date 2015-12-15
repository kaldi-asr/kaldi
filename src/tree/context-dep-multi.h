// tree/context-dep.h

// Copyright 2009-2011  Microsoft Corporation
//           2015       Hainan Xu

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

#ifndef KALDI_TREE_CONTEXT_DEP_MULTI_H_
#define KALDI_TREE_CONTEXT_DEP_MULTI_H_

#include "itf/context-dep-itf.h"
#include "tree/event-map.h"
#include "matrix/matrix-lib.h"
#include "tree/cluster-utils.h"
#include "tree/context-dep.h"
#include "hmm/hmm-topology.h"

namespace kaldi {

class ContextDependencyMulti: public ContextDependencyInterface {
 public:
  virtual int32 ContextWidth() const { return N_; }
  virtual int32 CentralPosition() const { return P_; }

  /// returns success or failure; outputs pdf to pdf_id
  virtual bool Compute(const std::vector<int32> &phoneseq,
                       int32 pdf_class, int32 *pdf_id) const;
  virtual int32 NumPdfs() const { // this routine could be simplified to return to_pdf_->MaxResult()+1.  we're a
    // bit more paranoid than that.
    if (!to_pdf_) return 0;
    EventAnswerType max_result = to_pdf_->MaxResult();
    if (max_result < 0 ) return 0;
    else return (int32) max_result+1;
  }
  virtual ContextDependencyInterface *Copy() const {
    vector<const EventMap*> trees;
    for (int i = 0; i < single_trees_.size(); i++) {
      trees.push_back(single_trees_[i]->Copy());
    }

    return new ContextDependencyMulti(N_, P_, trees, topo_);
  }

  /// Read context-dependency object from disk; throws on error
  void Read(std::istream &is, bool binary);

  // Constructor with no arguments; will normally be called
  // prior to Read()
  ContextDependencyMulti(): N_(0), P_(0), to_pdf_(NULL) { }

  // Constructor takes ownership of pointers.
  ContextDependencyMulti(int32 N, int32 P,
                         const vector<const EventMap*> &single_trees,
                         const HmmTopology &topo):
      N_(N), P_(P), single_trees_(single_trees), topo_(topo) {
    BuildVirtualTree();
  }

  // Constructor for when individual trees don't have the same N and Ps
  // We take the owndership of the input trees and also modify these trees
  // hence not "const EventMap*"
  ContextDependencyMulti(const vector<std::pair<int32, int32> > &NPs,
                         const vector<EventMap*> &single_trees,
                         const HmmTopology &topo);

  void Write(std::ostream &os, bool binary) const;

  void WriteVirtualTree(std::ostream &os, bool binary) const;
  void WriteMapping(std::ostream &os, bool binary) const;

  // caller not taking ownership of the tree
  void GetVirtualTreeAndMapping(EventMap** tree,
                                unordered_map<int32, vector<int32> > *m) {
    // *tree = to_pdf_->Copy();
    *tree = to_pdf_;
    *m = mappings_;
  }

  // caller taking ownership of the pointer
  EventMap* GetTree(size_t index) const {
    return single_trees_[index]->Copy();
  }

  ~ContextDependencyMulti() {
    for (int i = 0; i < single_trees_.size(); i++) {
      delete single_trees_[i];
    }

    delete to_pdf_;
  }

  const EventMap &ToPdfMap() const { return *to_pdf_; }

  /// GetPdfInfo returns a vector indexed by pdf-id, saying for each pdf which
  /// pairs of (phone, pdf-class) it can correspond to.  (Usually just one).
  /// c.f. hmm/hmm-topology.h for meaning of pdf-class.

  void GetPdfInfo(const std::vector<int32> &phones,  // list of phones
                  const std::vector<int32> &num_pdf_classes,  // indexed by phone,
                  std::vector<std::vector<std::pair<int32, int32> > > *pdf_info)
      const;


 private:
  int32 N_;  //
  int32 P_;
  EventMap *to_pdf_;  // owned here. This is the virtual tree
  vector<const EventMap*> single_trees_; // single trees, owned here
  HmmTopology topo_;
  unordered_map<int32, vector<int32> > mappings_;

  void BuildVirtualTree();
  void ConvertTreeContext(int32 old_P, int32 new_P, EventMap* new_tree);
                          

  KALDI_DISALLOW_COPY_AND_ASSIGN(ContextDependencyMulti);
};

}  // namespace Kaldi


#endif

