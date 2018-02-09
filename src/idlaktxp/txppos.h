// idlaktxp/txppos.h

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

#ifndef KALDI_IDLAKTXP_TXPPOS_H_
#define KALDI_IDLAKTXP_TXPPOS_H_

// This file defines greedy part of speech tagger class which
// determines pos tags for each normalised word and the tag set
// class which hold definitions of tag types

#include <map>
#include <utility>
#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"

namespace kaldi {

struct TxpPosRgx;

/// Whether a tagger regular expression is a whole word
/// a prefix or a suffix
enum TXPPOSRGX_POS {TXPPOSRGX_POS_WORD = 0,
                    TXPPOSRGX_POS_PREFIX = 1,
                    TXPPOSRGX_POS_SUFFIX = 2};

/// An array of tagger regular expressions
typedef std::vector<TxpPosRgx> RgxVector;

/// Part of speech tagger
/// see /ref idlaktxp_pos
class TxpPos: public TxpXmlData {
 public:
  explicit TxpPos() {}
  ~TxpPos() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "postags", name);
  }
  bool Parse(const std::string &tpdb);
  /// Return the part of speech for word current with previous
  /// context POS prev
  const char* GetPos(const char* prev, const char* current);

 private:
  void StartElement(const char* name, const char** atts);
  /// For loading pattern tagger
  std::string word_;
  // Tagger data
  /// Most common part of speech (default pos)
  std::string most_common_;
  /// Array of regular expression taggers. Note this doesn't use pcre
  /// because the regexs are only ever whole words, prefixes or suffixes
  RgxVector rgxtagger_;
  /// unigram and POS_word bigram patterns.
  LookupMap patterntagger_;
};

/// Holds the POS regex information and the tag it returns
struct TxpPosRgx {
 public:
  /// Whether whole word, prefix, suffix
  enum TXPPOSRGX_POS pos;
  /// The characters
  std::string pattern;
  /// The resulting tag
  std::string tag;
};

/// Part of speech sets
/// see /ref idlaktxp_pos
class TxpPosSet: public TxpXmlData {
 public:
  explicit TxpPosSet() {}
  ~TxpPosSet() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "posset", name);
  }
  bool Parse(const std::string &tpdb);
  /// Return the part of speech for word current with previous
  /// context POS prev
  const char* GetPosSet(const char* pos);

 private:
  void StartElement(const char* name, const char** atts);
  /// unigram and POS_word bigram patterns.
  LookupMap posset_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPPOS_H_
