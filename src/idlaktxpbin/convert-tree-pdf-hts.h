// idlaktxpbin/convert-tree-pdf-hts.h

// Copyright 2015 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#ifndef KALDI_IDLAKTXPBIN_CONVERT_TREE_PDF_HTS_H_
#define KALDI_IDLAKTXPBIN_CONVERT_TREE_PDF_HTS_H_

// Used to read in the context setup information xml

#include <string>
#include <pugixml.hpp>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "idlaktxp/idlaktxp.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/event-map.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "matrix/matrix-lib.h"
#include "fst/fstlib.h"

namespace kaldi {

class ConvertCex;
class TreeNode;

///Array to hold context extraction features and information
///held by kaldi index (kaldi phone context followed by fuill context)
typedef std::vector<ConvertCex> ConvertCexVector;
typedef std::vector<TreeNode *> TreeNodeVector;

class TreeNode {
 public:
  explicit TreeNode(const std::string &htsquestionname,
                    const std::string &htsquestion) {
    htsquestionname_ = htsquestionname;
    htsquestion_ = htsquestion;
    yes_ = -1;
    no_ = -1;
    // defaults are non-terminal branch
    yesval_ = -1;
    noval_ = -1;
  }
  void SetParent(bool yes, int32 idx) {
    if (yes) yes_ = idx;
    else no_ = idx;
  }
  void SetVal(bool yes, int32 val) {
    if (yes) yesval_ = val;
    else noval_ = val;
  }
  const std::string & GetHtsName() {
    return htsquestionname_;
  }
  const std::string & GetHtsQuestion() {
    return htsquestion_;
  }
  bool IsTerminalYes() {
    if (yesval_ >= 0) return true;
    return false;
  }
  bool IsTerminalNo() {
    if (noval_ >= 0) return true;
    return false;
  }
  int32 GetYes() {
    if (yesval_ >= 0) return yesval_;
    return yes_;
  }
  int32 GetNo() {
    if (noval_ >= 0) return noval_;
    return no_;
  }
    
 private:
  std::string htsquestionname_;
  std::string htsquestion_;
  int32 yes_;
  int32 no_;
  int32 yesval_;
  int32 noval_;
};
  
class ConvertCex {
 public:
  explicit ConvertCex() {}
  void Init(const std::string &name,
            const std::string &prvdelim,
            const std::string &pstdelim,
            bool isinteger) {
    name_ = name;
    prvdelim_ = prvdelim;
    pstdelim_ = pstdelim;
    isinteger_ = isinteger;
    vals_.clear();
    //std::cout << vals_.size() << "\n";
  }
  int32 AddVal(const std::string &val) {
    vals_.push_back(val);
    return vals_.size();
  }
  const std::string &LookupVal(int32 index) const {
    if (index < 0 || index >= vals_.size()) return empty;
    return vals_.at(index);
  }
  void SetVal(int32 index, const std::string &val) {
    if (index < 0) index = 0;
    if (index >= vals_.size()) vals_.resize(index + 1);
    //std::cout << vals_.size() << " " << index << " " << val << "\n";
    vals_.at(index) = val;
  }
  const std::string &GetName() const {return name_;}
  bool GetIsinteger() const {return isinteger_;}
  const std::string &GetPrvdelim() const {return prvdelim_;}
  const std::string &GetPstdelim() const {return pstdelim_;}
    
 private:
  /// Context name
  std::string name_;
  /// Whether a string or integer feature
  bool isinteger_;
  /// HTS delimeter string before context
  std::string prvdelim_;
  /// HTS delimeter string after context
  std::string pstdelim_;
  /// integer to value lookup if required
  StringVector vals_;
  /// empty string
  std::string empty;
};

}

#endif  //KALDI_IDLAKTXPBIN_CONVERT_TREE_PDF_HTS_H_
