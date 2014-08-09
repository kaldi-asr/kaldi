// bin/extract-ctx.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#include <vector>
#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-vector.h"
#include "tree/build-tree.h"
#include "tree/clusterable-classes.h"
#include "tree/context-dep.h"
#include "tree/build-tree-questions.h"
#include "fst/fstlib.h"

using namespace kaldi;

using std::vector;

// Generate a string representation of the given EventType;  the symtable is
// optional, so is the request for positional symbols (tri-phones: 0-left,
// 1-center, 2-right.
static std::string EventTypeToString(EventType &e,
                                     fst::SymbolTable *phones_symtab,
                                     bool addpos) {
  // make sure it's sorted so that the kPdfClass is the first element!
  std::sort(e.begin(), e.end());
  
  // first plot the pdf-class
  std::stringstream ss;
  ss << e[0].second;
  for (size_t i = 1; i < e.size(); ++i) {
    ss << " ";
    if (addpos)
      ss << (i-1) << ":";
    
    if (phones_symtab == NULL)
      ss << e[i].second;
    else {
      std::string phn =
        phones_symtab->Find(static_cast<kaldi::int64>(e[i].second));
      if (phn.empty()) {
        // in case we can't resolve the symbol, plot the ID
        KALDI_WARN << "No phone found for ID " << e[i].second;
        ss << e[i].second;
      } else
        ss << phn;
    }
  }
  return ss.str();
}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    const char *usage =
      "Given the tree stats and the resulting tree, output a mapping of phones\n"
      "in context (and pdf-class) to the pdf-id.  This can be used to link the\n"
      "acoustic model parameters to their phonetic meaning.  Outputs lines such as\n"
      "  \"<pdf-id> <pdf-class> <left> <center> <right>\" in case of tri-phones\n"
      "e.g.: \n"
      " extract-ctx treeacc tree\n"
      " extract-ctx --mono 48 tree\n";
    
    ParseOptions po(usage);
    
    std::string fsymboltab;
    bool addpos = false;
    bool mono = false;
    std::string silphones = "1,2,3";
    int32 silpdfclasses = 5;
    int32 nonsilpdfclasses = 3;
    
    po.Register("mono", &mono,
                "Assume mono-phone tree;  instead of tree stats, specify highest id");
    po.Register("sil-phones", &silphones,
                "[only for --mono] Comma separated list of silence phones");
    po.Register("sil-pdf-classes", &silpdfclasses,
                "[only for --mono] Number of pdf-classes for silence phones");
    po.Register("non-sil-pdf-classes", &nonsilpdfclasses,
                "[only for --mono] Number of pdf-classes for non-silence phones");
    po.Register("symbol-table", &fsymboltab,
                "Specify phone sybol table for readable output");
    po.Register("add-position-indicators", &addpos,
                "Add position indicators for phonemes");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    // read symtab if available
    fst::SymbolTable *phones_symtab = NULL;
    if (fsymboltab.length() > 0) {
      KALDI_LOG << "Reading symbol table from " << fsymboltab;
      std::ifstream is(fsymboltab.c_str());
      phones_symtab = ::fst::SymbolTable::ReadText(is, fsymboltab);
      if (!phones_symtab)
        KALDI_ERR << "Could not read phones symbol table file "<< fsymboltab;
    }
    
    // read the tree, get all the leaves
    ContextDependency ctx_dep;
    ReadKaldiObject(po.GetArg(2), &ctx_dep);
    const EventMap &map = ctx_dep.ToPdfMap();
    
    // here we have to do different things for mono and tri+ trees
    if (mono) {
      // A mono-phone tree is not actually a real tree.  We test for EventTypes
      // that have the central phone and the possible pdf-classes
      
      int32 maxs = atoi(po.GetArg(1).c_str());
      if (phones_symtab != NULL) {
        size_t ns = phones_symtab->NumSymbols();
        if (maxs != (int32) (ns-1)) {
          KALDI_WARN << "specified highest symbol (" << maxs
            << ") not equal to size of symtab (" << (ns-1) << "), adjusting ";
          maxs = (ns-1);
        }
      }
      
      // parse silphones
      std::set<int32> silset;
      
      std::string::size_type i1 = 0, i2;
      do {
        i2 = silphones.find(',', i1);
        silset.insert(atoi(silphones.substr(i1, i2 - i1).c_str()));
        KALDI_LOG << "silphone: " << silphones.substr(i1, i2 - i1);
        if (i2 == std::string::npos)
          break;
        i1 = i2 + 1;
      } while (true);
                      
      
      // now query each phone (ignore <eps> which is 0)
      for (int32 p = 1; p <= maxs; ++p) {
        int32 mpdf = (silset.find(p) == silset.end() ?
                        nonsilpdfclasses :
                        silpdfclasses);
        
        for (int i = 0; i < mpdf; ++i) {
          EventType et;
          et.push_back(std::pair<EventKeyType, EventValueType>(kPdfClass, i));
          et.push_back(std::pair<EventKeyType, EventValueType>(0, p));
          
          EventAnswerType ans;
          if (map.Map(et, &ans)) {
            std::cout << ans << " "
              << EventTypeToString(et, phones_symtab, addpos)
              << std::endl;
          } else {
            KALDI_WARN << "Unable to get pdf-id for stats "
              << EventTypeToString(et, phones_symtab, addpos);
          }

        }
      }
      
    } else {
      // for tri+ trees, read the tree stats;  this gives us basically all
      // phones-in-context that may be linked to an individual model
      // (in practice, many of them will be shared, but we plot them anyways)
      
      // build-tree-questions.h:typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType
      BuildTreeStatsType stats;
      {
        bool binary_in;
        GaussClusterable gc;  // dummy needed to provide type.
        Input ki(po.GetArg(1), &binary_in);
        ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
      }
      KALDI_LOG << "Number of separate statistics is " << stats.size();
      
      // typedef std::vector<std::pair<EventKeyType,EventValueType> > EventType
      
      // now, for each tree stats element, query the tree to get the pdf-id
      for (size_t i = 0; i < stats.size(); ++i) {
        EventAnswerType ans;
        if (map.Map(stats[i].first, &ans)) {
          std::cout << ans << " "
            << EventTypeToString(stats[i].first, phones_symtab, addpos)
            << std::endl;
        } else {
          KALDI_WARN << "Unable to get pdf-id for stats "
            << EventTypeToString(stats[i].first, phones_symtab, addpos);
        }
      }
    }
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


