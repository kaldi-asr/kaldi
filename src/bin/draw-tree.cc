// bin/draw-tree.cc

// Copyright 2012 Vassil Panayotov

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

#include "tree/tree-renderer.h"

void MakeEvent(std::string &qry, fst::SymbolTable *phone_syms,
               kaldi::EventType **query)
{
  using namespace kaldi;

  EventType *query_event = new EventType();
  size_t found, old_found = 0;
  EventKeyType key = kPdfClass; // this code in fact relies on kPdfClass = -1
  while ((found = qry.find('/', old_found)) != std::string::npos) {
    std::string valstr = qry.substr(old_found, found - old_found);
    EventValueType value;
    if (key == kPdfClass) {
      value = static_cast<EventValueType>(atoi(valstr.c_str()));
      if (value < 0) { // not valid pdf-class
        KALDI_ERR << "Bad query: invalid pdf-class ("
                  << valstr << ')' << std::endl << std::endl;
      }
    }
    else {
      value = static_cast<EventValueType>(phone_syms->Find(valstr.c_str()));
      if (value == fst::SymbolTable::kNoSymbol) {
        KALDI_ERR << "Bad query: invalid symbol ("
                  << valstr << ')' << std::endl << std::endl;
      }
    }
    query_event->push_back(std::make_pair(key++, value));
    old_found = found + 1;
  }
  std::string valstr = qry.substr(old_found);
  EventValueType value = static_cast<EventValueType>(phone_syms->Find(valstr.c_str()));
  if (value == fst::SymbolTable::kNoSymbol) {
    KALDI_ERR << "Bad query: invalid symbol ("
              << valstr << ')' << std::endl << std::endl;
  }
  query_event->push_back(std::make_pair(key, value));

  *query = query_event;
}

int main(int argc, char **argv) {
  using namespace kaldi;
  try {
    std::string qry;
    bool use_tooltips = false;
    bool gen_html = false;

    const char *usage =
        "Outputs a decision tree description in GraphViz format\n"
        "Usage: draw-tree [options] <phone-symbols> <tree>\n"
        "e.g.: draw-tree phones.txt tree | dot -Gsize=8,10.5 -Tps | ps2pdf - tree.pdf\n";
    
    ParseOptions po(usage);
    po.Register("query", &qry,
                "a query to trace through the tree"
                "(format: pdf-class/ctx-phone1/.../ctx-phoneN)");
    po.Register("use-tooltips", &use_tooltips, "use tooltips instead of labels");
    po.Register("gen-html", &gen_html, "generates HTML boilerplate(useful with SVG)");
    po.Read(argc, argv);
    if (gen_html) {
      std::cout << "<html>\n<head><title>Decision Tree</tree></head>\n"
                << "<body><object data=\"tree.svg\" type=\"image/svg+xml\"/>"
                << "</body>\n</html>\n";
      return 0;
    }
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      return -1;
    }
    std::string phnfile = po.GetArg(1);
    std::string treefile = po.GetArg(2);

    fst::SymbolTable *phones_symtab = NULL;
    {
        std::ifstream is(phnfile.c_str());
        phones_symtab = ::fst::SymbolTable::ReadText(is, phnfile);
        if (!phones_symtab)
            KALDI_ERR << "Could not read phones symbol table file "<< phnfile;
    }

    EventType *query = NULL;
    if (!qry.empty())
      MakeEvent(qry, phones_symtab, &query);

    TreeRenderer *renderer = NULL;
    {
      bool binary;
      Input ki(treefile, &binary);
      renderer = new TreeRenderer(ki.Stream(), binary, std::cout,
                                  *phones_symtab, use_tooltips);
      renderer->Render(query);
    }

    if (renderer) delete renderer;
    if (query) delete query;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
