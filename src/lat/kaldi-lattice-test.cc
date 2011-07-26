// lat/kaldi-lattice.cc

// Copyright 2009-2011     Microsoft Corporation

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


#include "lat/kaldi-lattice.h"
#include "fstext/rand-fst.h"


namespace kaldi {

CompactLattice *RandCompactLattice() {
  Lattice *fst = fst::RandPairFst<LatticeArc>();
  CompactLattice *cfst = new CompactLattice;
  ConvertLattice(*fst, cfst);
  delete fst;
  return cfst;
}

Lattice *RandLattice() {
  Lattice *fst = fst::RandPairFst<LatticeArc>();
  return fst;
}

void TestLatticeTable() {
  CompactLatticeWriter writer("ark:tmpf");
  int N = 10;
  std::vector<CompactLattice*> lat_vec(N);
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    CompactLattice *fst = RandCompactLattice();
    lat_vec[i] = fst;
    writer.Write(key, *fst);
  }
  writer.Close();

  RandomAccessCompactLatticeReader reader("ark:tmpf");  
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    const CompactLattice &fst = reader.Value(key);
    KALDI_ASSERT(fst::Equal(fst, *(lat_vec[i])));
    delete lat_vec[i];
  }
}


void TestLatticeTable2() { // text mode.
  CompactLatticeWriter writer("ark,t:tmpf");
  int N = 10;
  std::vector<CompactLattice*> lat_vec(N);
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    CompactLattice *fst = RandCompactLattice();
    lat_vec[i] = fst;
    writer.Write(key, *fst);
  }
  writer.Close();

  RandomAccessCompactLatticeReader reader("ark:tmpf");
  CompactLatticeWriter writer2("ark,t:-");
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    const CompactLattice &fst = reader.Value(key);
    writer2.Write(key, fst);
    //KALDI_ASSERT(fst::Equal(fst, *(lat_vec[i])));
    delete lat_vec[i];
  }
}

} // end namespace kaldi

int main() {
  using namespace kaldi;
  TestLatticeTable();
  TestLatticeTable2();
  std::cout << "Test OK\n";
}



