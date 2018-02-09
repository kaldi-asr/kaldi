// idlaktxp/txplts.h

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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

#ifndef KALDI_IDLAKTXP_TXPLTS_H_
#define KALDI_IDLAKTXP_TXPLTS_H_

// This file defines the cart letter to sound system

#include <map>
#include <utility>
#include <vector>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"
#include "idlaktxp/txplexicon.h"

namespace kaldi {

struct TxpLtsNode;
struct TxpLtsTree;

/// Array of nodes used in the cart tree
typedef std::vector<TxpLtsNode> TxpLtsNodes;
/// Lookup for letter to tree
typedef std::map<std::string, TxpLtsTree> LtsMap;
/// letter/tree pair
typedef std::pair<std::string, TxpLtsTree> LtsItem;

/// Contains a cart tree for each letter
/// Questions are left and right context letters
/// Trees are typically built from a source lexicon using speechtools
/// wagon
class TxpLts: public TxpXmlData {
 public:
  explicit TxpLts() {}
  ~TxpLts() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "ccart", name);
  }
  /// Given a word return the result of looking up the pronciation of
  /// of each letter in a cart tree
  int GetPron(const std::string &word, TxpLexiconLkp* lkp);

 private:
  void StartElement(const char* name, const char** atts);
  /// Lookup from letter to cart tree
  LtsMap ltslkp_;
  /// holds parser status of current letter
  std::string ltr_;
};

/// A cart tree
struct TxpLtsTree {
  /// the index of the root node in the node array
  int32 root;
  /// Array of terminal and non-terminal nodes
  TxpLtsNodes nodes;
};

/// Structure for terminal and non-terminal nodes
struct TxpLtsNode {
  /// if non-terminal offset of context letter
  int32 pos;
  /// if non-terminal the value of the context letter
  std::string posval;
  /// if terminal the value of the phone
  std::string val;
  /// if non-terminal the index of the node to go to if the
  /// context matches
  int32 yes;
  /// if non-terminal the index of the node to go to if the
  /// context does not match
  int32 no;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPLTS_H_
