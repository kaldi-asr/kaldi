// dec-wrap/dec-wrap-util.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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
#include <string>
#include "dec-wrap/dec-wrap-utils.h"
#include "lat/kaldi-lattice.h"
#include "fstext/fstext-utils.h"
#include "fstext/lattice-utils-inl.h"

// DEBUG
#include "lat/lattice-functions.h"

namespace kaldi {

void MovePostToArcs(fst::VectorFst<fst::LogArc> * lat, 
                          const std::vector<double> &alpha,
                          const std::vector<double> &beta) {
  using namespace fst;
  typedef typename LogArc::StateId StateId;
  StateId num_states = lat->NumStates();
  for (StateId i = 0; i < num_states; ++i) {
    for (MutableArcIterator<VectorFst<LogArc> > aiter(lat, i); 
        !aiter.Done();
         aiter.Next()) {
      LogArc arc = aiter.Value();
      StateId j = arc.nextstate;
      // w(i,j) = alpha(i) * w(i,j) * beta(j) / (alpha(i) * beta(i))
      // w(i,j) = w(i,j) * beta(j) / beta(i)
      double orig_w = ConvertToCost(arc.weight);
      double numer = orig_w + -beta[j];
      KALDI_VLOG(3) << "arc(" << i << ',' << j << ')' << std::endl << 
        "orig_w:" << orig_w << " beta[j=" << j << "]:" << -beta[j] << 
        " beta[i=" << i << "]:" << -beta[i] << " numer:" << numer << std::endl;
      double new_w = numer - (-beta[i]);
      KALDI_VLOG(3) << "arc orig: " << orig_w << " new: " << new_w << std::endl;
      arc.weight = LogWeight(new_w);

      aiter.SetValue(arc);
    }
  }
}

double CompactLatticeToWordsPost(CompactLattice &clat, fst::VectorFst<fst::LogArc> *pst) {
#ifdef DEBUG_COMPACT_LAT
  std::string lattice_wspecifier("ark:|gzip -c > after_getLattice.gz");
  CompactLatticeWriter compact_lattice_writer;
  compact_lattice_writer.Open(lattice_wspecifier);
  compact_lattice_writer.Write("unknown", clat);
  compact_lattice_writer.Close();

  CompactLattice clat_best_path;
  CompactLatticeShortestPath(clat, &clat_best_path);  // A specialized
  Lattice best_path;
  ConvertLattice(clat_best_path, &best_path);
  std::vector<int32> alignment;
  std::vector<int32> words;
  LatticeWeight weight;
  GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
  std::cerr << "CompactLattice best path: cost:"
            << weight.Value1() << " + " << weight.Value2() << " = "
            << (weight.Value1() + weight.Value2()) << std::endl;
  std::cerr << "CompactLattice best path: words:";
  for (size_t i = 0; i < words.size(); ++i)
    std::cerr << words[i] << " ";
  std::cerr << std::endl;
#endif // DEBUG_COMPACT_LAT

  {
    Lattice lat;
    fst::VectorFst<fst::StdArc> t_std;
    RemoveAlignmentsFromCompactLattice(&clat); // remove the alignments
    ConvertLattice(clat, &lat); // convert to non-compact form.. no new states
#ifdef DEBUG_CONVERT_LAT
    LatticeWriter lattice_writer;
    std::string lattice_wspecifier("ark:|gzip -c > after_convertLattice_lat.gz");
    compact_lattice_writer.Open(lattice_wspecifier);
    compact_lattice_writer.Write("unknown", clat);
    compact_lattice_writer.Close();
#endif // DEBUG_CONVERT_LAT
    ConvertLattice(lat, &t_std); // this adds up the (lm,acoustic) costs
#ifdef DEBUG_CONVERT_TROP
    {
      std::ofstream logfile;
      logfile.open("after_convert_trop.fst");
      t_std.Write(logfile, fst::FstWriteOptions());
      logfile.close();
    }
#endif // DEBUG_CONVERT_TROP
    fst::Cast(t_std, pst);  // reinterpret the inner implementations
  }
#ifdef DEBUG_CAST
  {
    std::ofstream logfile;
    logfile.open("after_cast.fst");
    pst->Write(logfile, fst::FstWriteOptions());
    logfile.close();
  }
#endif // DEBUG_CAST
  fst::Project(pst, fst::PROJECT_OUTPUT);


  fst::Minimize(pst);
#ifdef DEBUG_MIN
  {
    std::ofstream logfile;
    logfile.open("after_minimize.fst");
    pst->Write(logfile, fst::FstWriteOptions());
    logfile.close();
  }
#endif // DEBUG_MIN

  fst::ArcMap(pst, fst::SuperFinalMapper<fst::LogArc>());
#ifdef DEBUG_FINAL
  {
    std::ofstream logfile;
    logfile.open("after_super_final.fst");
    pst->Write(logfile, fst::FstWriteOptions());
    logfile.close();
  }
#endif // DEBUG_FINAL

  double tot_prob;
  std::vector<double> alpha, beta;
  fst::TopSort(pst);
  tot_prob = ComputeLatticeAlphasAndBetas(*pst, &alpha, &beta);
  MovePostToArcs(pst, alpha, beta);
#ifdef DEBUG_POST
  for (size_t i = 0; i < alpha.size(); ++i) {
    std::cerr << "a[" << i << "] = " << alpha[i] << " beta[" << i << "] = "
      << beta[i] << std::endl;
  }
  {
    std::ofstream logfile;
    logfile.open("after_post.fst");
    pst->Write(logfile, fst::FstWriteOptions());
    logfile.close();
  }
#endif // DEBUG_POST

  return tot_prob;
}


fst::Fst<fst::StdArc> *ReadDecodeGraph(std::string filename) {
  // read decoding network FST
  Input ki(filename); // use ki.Stream() instead of is.
  if (!ki.Stream().good()) KALDI_ERR << "Could not open decoding-graph FST "
                                      << filename;

  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    KALDI_ERR << "Reading FST: error reading FST header.";
  }
  if (hdr.ArcType() != fst::StdArc::Type()) {
    KALDI_ERR << "FST with arc type " << hdr.ArcType() << " not supported.\n";
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = NULL;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else {
    KALDI_ERR << "Reading FST: unsupported FST type: " << hdr.FstType();
  }
  if (decode_fst == NULL) { // fst code will warn.
    KALDI_ERR << "Error reading FST (after reading header).";
    return NULL;
  } else {
    return decode_fst;
  }
}


void PrintPartialResult(const std::vector<int32>& words,
                        const fst::SymbolTable *word_syms,
                        bool line_break) {
  KALDI_ASSERT(word_syms != NULL);
  for (size_t i = 0; i < words.size(); i++) {
    std::string word = word_syms->Find(words[i]);
    if (word == "")
      KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
    std::cout << word << ' ';
  }
  if (line_break)
    std::cout << "\n\n";
  else
    std::cout.flush();
}


// converts  phones to vector representation
std::vector<int32> phones_to_vector(const std::string & s) {
  std::vector<int32> return_phones;
  if (!SplitStringToIntegers(s, ":", false, &return_phones))
      KALDI_ERR << "Invalid silence-phones string " << s;
  if (return_phones.empty())
      KALDI_ERR << "No silence phones given!";
  return return_phones;
} // phones_to_vector

} // namespace kaldi
