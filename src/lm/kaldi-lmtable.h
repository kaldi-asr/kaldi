// lm/kaldi-lmtable.h
// Copyright Gilles Boulianne.

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_LM_KALDI_LMTABLE_H_
#define KALDI_LM_KALDI_LMTABLE_H_

// To use IRSTLM toolkit, use #define HAVE_IRSTLM
// otherwise there is limited support (reading of arpa files)
// provided by kaldi-lmtable.cc

#include <fstream>
#include <vector>
#include <string>

#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;

#ifndef HAVE_IRSTLM
#else
#include "irstlm/lmtable.h"
#include "irstlm/n_gram.h"
#endif

#include "fst/fstlib.h"
#include "fst/fst-decl.h"
#include "fst/arc.h"
#include "base/kaldi-common.h"
#include "util/stl-utils.h"

#ifdef _MSC_VER
#  define STRTOF(cur_cstr, end_cstr) static_cast<float>(strtod(cur_cstr, end_cstr));
#else
#  define STRTOF(cur_cstr, end_cstr) strtof(cur_cstr, end_cstr);
#endif


namespace kaldi {
/// @addtogroup LanguageModel
/// @{
/// @file kaldi-lmtable.h
/**
  * @brief Definition of internal representation for language model.
  *
  * Provides a unified interface to various toolkits such as IRSTLM.
  * Also contains a basic implementation for reading ARPA files which
  * does not require an external library.
*/


/// @brief Helper methods to convert toolkit internal representations into FST.
class LmFstConverter {
  typedef fst::StdArc::Weight LmWeight;
  typedef fst::StdArc::StateId StateId;
  
  typedef unordered_map<StateId, StateId> BkStateMap;
  typedef unordered_map<std::string, StateId, StringHasher> HistStateMap;

 public:

  LmFstConverter() : use_natural_log_(true) {}

  ~LmFstConverter() {}

  void UseNaturalLog(bool use_natural) { use_natural_log_ = use_natural; }

  void AddArcsForNgramProb(int ilev,
                           int maxlev,
                           float prob,
                           float bow,
                           std::vector<string> &ngs,
                           fst::StdVectorFst *fst,
                           const string startSent,
                           const string endSent);

  float ConvertArpaLogProbToWeight(float lp) {
    if ( use_natural_log_ ) {
      // convert from arpa base 10 log to natural base, then to cost
      return -2.302585*lp;
    } else {
      // keep original base but convert to cost
      return -lp;
    }
  }

  bool IsFinal(fst::StdVectorFst *pfst,
               StateId s) {
    return(pfst->Final(s) != fst::StdArc::Weight::Zero());
  }

  void ConnectUnusedStates(fst::StdVectorFst *pfst);

 private:
  StateId AddStateFromSymb(const std::vector<string> &ngramString,
                            int kstart,
                            int kend,
                            fst::StdVectorFst *pfst,
                            bool &newlyAdded);

  StateId FindState(const std::string str) {
    HistStateMap::const_iterator it = histState_.find(str);
     if (it == histState_.end()) {
       return -1;
     }
     return it->second;
  }

  bool use_natural_log_;
  BkStateMap bkState_;
  HistStateMap histState_;
};

#ifndef HAVE_IRSTLM

/** @brief Basic Kaldi implementation for reading ARPA format files.
  *
  * It does not rely on any external toolkit
  * but only supports standard ARPA format language model files.
*/
class LmTable {
 public:
  LmTable() { conv_ = new LmFstConverter; }
  ~LmTable() { if (conv_) delete conv_; }

  bool ReadFstFromLmFile(std::istream &istrm,
                         fst::StdVectorFst *pfst,
                         bool useNaturalLog,
                         const string startSent,
                         const string endSent);
 private:
  LmFstConverter *conv_;
};

#else

// special value for pruned iprobs
#define NOPROB (static_cast<float>(-1.329227995784915872903807060280344576e36))

/// @brief IRSTLM implementation that inherits from IRSTLM table class
/// in order to access internal data needed to create an FST.
class LmTable : public lmtable {
 public:
  LmTable() { conv_ = new LmFstConverter; }
  ~LmTable() { if (conv_) delete conv_; }

  /// in this implementation, needed functions come from parent class, e.g.
  ///   table_entry_pos_t wdprune(float *thr, int aflag = 0);
  ///   void load(std::istream& inp, ...);
  bool ReadFstFromLmFile(std::istream &istrm,
                       fst::StdVectorFst *pfst,
                        bool useNaturalOpt,
                       const string startSent,
                       const string endSent);
 private:

  /// Method specific to the IRSTLM implementation.
  void DumpStart(ngram ng,
                 fst::StdVectorFst *pfst,
                 const string startSent,
                 const string endSent);
  /// Method specific to the IRSTLM implementation.
  void DumpContinue(ngram ng,
                    int ilev, int elev,
                    table_entry_pos_t ipos, table_entry_pos_t epos,
                    fst::StdVectorFst *pfst,
                    const string startSent, const string endSent);

  LmFstConverter *conv_;
};

#endif
/// @} end of "LanguageModel"
}  // end namespace kaldi


#endif  // KALDI_LM_KALDI_LMTABLE_H_
