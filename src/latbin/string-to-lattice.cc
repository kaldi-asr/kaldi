// latbin/string-to-lattice.cc

// Copyright 2011 Gilles Boulianne
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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include <stdexcept>

using namespace kaldi;

// add the string contained in inpline to the current transducer
// starting at initial state
std::string ReadTxtLine( const string &inpline, Lattice *pfst) {

  KALDI_ASSERT(pfst);
  std::string key;
  fst::StdArc::StateId src, dst;
  pfst->AddState();  // adding state 0 assuming empty fst
  pfst->SetStart(0);
  src = pfst->Start();
  // this will split on white spaces only
  fst::StdArc::Label label;  // Have a buffer string
  std::stringstream ss(inpline);  // Insert the string into a stream
  ss >> key;
  //cerr << "ReadTxtLine: key = ["<<key<<"]"<<endl;
   while (ss >> label) {
    // add labels to symbol tables
    dst = pfst->AddState();
    pfst->AddArc(src, LatticeArc(label, label, LatticeWeight(0.0,0.0), dst));
    //cerr << "  added word " << label << " from state " << src;
    //cerr << " to state " << dst <<endl;
    src = dst;
  }
  pfst->SetFinal(dst, LatticeArc::Weight::One());
  return key;
}

// convert from Lattice to unweighted FST and sort it

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::uint32 uint32;
    typedef kaldi::uint64 uint64;
    
    const char *usage =
        "Read strings at standard input and writes them out as a lattice.\n"
        "Usage: string-to-lattice lattice-wspecifier\n"
        " e.g.: echo \"key 1032 3034 2039\" | string-to-lattice ark:1.lats\n";
        
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs()!= 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_wspecifier = po.GetArg(1);

    // Will write as lattice     
    LatticeWriter lattice_writer(lats_wspecifier);

    string inpline;
    std::string key;
    int keynum = 0;
    while (getline(std::cin, inpline) && !std::cin.eof()) {

      Lattice *lat = new Lattice();

#ifndef TEST
      // convert to an FST
      //cerr << "Read line ["<<inpline<<endl;
      key = ReadTxtLine(inpline, lat);
      
#else     
      Lattice *lat1 = fst::RandPairFst<LatticeArc>();
      fst::RandGen(*lat1, lat);
      std::stringstream ss;
      ss << keynum;
      key = ss.str();
#endif

      std::string msg((lat->Start() == fst::kNoStateId) ? "no" : "some");
      //cerr << "Lattice has "<<msg<<" initial state"<<endl;
      // write lattice
      lattice_writer.Write(key, *lat);
      //cerr << "Writing lattice with key "<<key<<endl;
      keynum++;
      delete lat;
    }
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
