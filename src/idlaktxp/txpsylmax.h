// idlaktxp/txpsylmax.h

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

#ifndef KALDI_IDLAKTXP_TXPSYLMAX_H_
#define KALDI_IDLAKTXP_TXPSYLMAX_H_

// This file contains functions which carry out maximal onset
// syllabification

#include <vector>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"
#include "idlaktxp/txputf8.h"

namespace kaldi {

/// Syllables subtype
enum TXPSYLMAX_TYPE {TXPSYLMAX_TYPE_CODA = 0,
                     TXPSYLMAX_TYPE_NUCLEUS = 1,
                     TXPSYLMAX_TYPE_ONSET = 2};

struct TxpSylItem;

/// Array to hold phones
/// To allow for laison this acts as a queue with the following words
/// pronunication loaded after the word being syllabified.
typedef std::vector<TxpSylItem> PhoneVector;

/// Applies maximal onset rule to create syllables
/// see \ref idlaktxp_syll
class TxpSylmax: public TxpXmlData {
 public:
  explicit TxpSylmax() : stress_(false), max_onset_(0), max_nucleus_(0) {}
  ~TxpSylmax() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "sylmax", name);
  }
  bool Parse(const std::string &tpdb);
  /// Apply maximal onset rules to the phones in the array
  void Maxonset(PhoneVector* pronptr);
  /// Convert phone array to a string based pronunciation for XML
  void Writespron(PhoneVector* pronptr, std::string* sylpron);
  /// Append a pronunciation onto a phone array
  int32 GetPhoneVector(const char* pron, PhoneVector* phonevectorptr);
  /// Decide if a phone is syllabic or not
  bool IsSyllabic(const char* phone);


 private:
  void StartElement(const char* name, const char** atts);
  /// Based on codas, onsets and nuclei set syllable boundaries in phone array
  int32 AddSylBound(PhoneVector *pron);
  /// Find if a valid nucleus begins here
  int32 FindNucleus(const PhoneVector &pron, int32 pos);
  /// Find if a valid onset ends here
  int32 FindOnset(const PhoneVector &pron, int32 pos);
  /// Append a phone starting at p into the phone array
  void InsertPhone(PhoneVector *phonevectorptr,
                   const char* p, int32 len, bool wbnd);
  /// Set nucleus pattern based on position and length in phone array
  /// return true if not blocked by a word boundary and sufficient items
  /// are available
  bool GetPhoneNucleusPattern(const PhoneVector &pron,
                              int32 pos, int32 len,
                              bool with_stress, std::string* pat);
  /// Set onset pattern based on position and length backwards in phone array
  /// return true if not blocked by a word boundary and sufficient items
  /// are available
  bool GetPhoneOnsetPattern(const PhoneVector &pron,
                            int32 pos, int32 len,
                            std::string* pat);
  /// phones which are syllabic
  StringSet syllabic_;
  /// valid nucleus patterns
  StringSet nuclei_;
  /// valid onset patterns
  StringSet onsets_;
  /// If stress true nucleus patterns must match correct stress
  bool stress_;
  /// longest onset pattern
  int32 max_onset_;
  /// longest nucleus pattern
  int32 max_nucleus_;
};

/// Structure to hold a phone together with its syllabic information
struct TxpSylItem {
 public:
  TxpSylItem () {}
  void Clear() {
    name = "";
    stress = "";
    type = TXPSYLMAX_TYPE_CODA;
    wrdb = false;
    sylb = false;
    codab = false;
    cross_word = false;
  }
  /// Phone name
  std::string name;
  /// Stress value
  std::string stress;
  /// syllable subtype
  enum TXPSYLMAX_TYPE type;
  /// word boundary following
  bool wrdb;
  /// syllable boundary following
  bool sylb;
  /// coda boundary following
  bool codab;
  /// liaison across word boundary
  bool cross_word;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPSYLMAX_H_
