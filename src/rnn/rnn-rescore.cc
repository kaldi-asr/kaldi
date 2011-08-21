// latbin/lattice-nbest.cc

// Copyright 2009-2011  Stefan Kombrink 

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

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <float.h>

#include "base/kaldi-common.h"
#include "base/kaldi-error.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fst/fst.h"
#include "fst/queue.h"
#include "fst/rmepsilon.h"
#include "fst/dfs-visit.h"
#include "lat/kaldi-lattice.h"

using namespace Eigen;
using namespace std;
using namespace kaldi;

// define if float or doubles are used
typedef float realn;
typedef MatrixXf MatrixXr;
typedef VectorXf VectorXr;
typedef kaldi::Matrix<realn> KaldiMatrix;
typedef kaldi::Vector<realn> KaldiVector;

class RNN{
  public:
    // override stupid ofst types
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

  private:
    MatrixXr V1_,U1_,W2_,Cl_; // weights
    VectorXr b1_,b2_,cl_b_;   // biases TODO not yet used!!!
    VectorXr h_,cl_,y_;       // activations

    map<int32,string> int2word_;                  // maps from rnn ints to word strings
    map<string,int32> word2int_;                  // maps from word strings to rnn ints
    map<int32,int32> intlat2intrnn;               // maps from ints in lattices to ints in RNN
    map<int32,int32> intrnn2intlat;               // maps from ints in RNN to ins in lattices
    vector<int32> int2class_;                     // mapping words (integer) to their classes
    map<int32,int32> class2minint_,class2maxint_; // determines the range of a class of words in the output layer

    // others...
    realn OOVPenalty_;

  protected:
    void KaldiToEigen(KaldiVector* in, VectorXr* out) {
      out->resize(in->Dim());
      for (int i=0;i<in->Dim();i++) (*out)(i)=(*in)(i);
      in->Resize(0,kUndefined);
    }

    void KaldiToEigen(KaldiMatrix* in, MatrixXr* out) {
      out->resize(in->NumRows(),in->NumCols());
      for (int row=0;row<in->NumRows();row++)
        for (int col=0;col<in->NumCols();col++)
          (*out)(row,col)=(*in)(row,col);
      in->Resize(0,0,kUndefined);
    }

  public:
    inline int32 VocabSize() const { return V1_.cols(); }
    inline int32 HiddenSize() const { return V1_.rows(); }
    inline int32 ClassSize() const { return Cl_.rows(); }

    inline bool IsIV(int32 w) const { return w>=0; }
    inline bool IsOOV(int32 w) const { return !IsIV(w); }

    realn OOVPenalty() const { return OOVPenalty_; }
    void SetOOVPenalty( realn oovp ) { OOVPenalty_=oovp; }

    void Read(istream& in, bool binary) {
      KaldiMatrix V1,U1,W2,Cl;
      KaldiVector b1,b2,cl_b;

      ExpectMarker(in,binary,"<rnnlm_v2.0>");

      ExpectMarker(in,binary,"<v1>"); in >> V1;
      V1.Transpose(); KaldiToEigen(&V1,&V1_);

      ExpectMarker(in,binary,"<u1>"); in >> U1;
      U1.Transpose(); KaldiToEigen(&U1,&U1_);

      ExpectMarker(in,binary,"<b1>"); in >> b1;
      KaldiToEigen(&b1,&b1_);

      ExpectMarker(in,binary,"<w2>"); in >> W2;
      W2.Transpose(); KaldiToEigen(&W2,&W2_);

      ExpectMarker(in,binary,"<b2>"); in >> b2;
      KaldiToEigen(&b2,&b2_);

      ExpectMarker(in,binary,"<cl>"); in >> Cl;
      Cl.Transpose(); KaldiToEigen(&Cl,&Cl_);

      ExpectMarker(in,binary,"<cl_b>"); in >> cl_b;
      KaldiToEigen(&cl_b,&cl_b_);

      ExpectMarker(in,binary,"<classes>");
      ReadIntegerVector(in,binary,&int2class_);

     // determine range for classes 
     // THIS ASSUMES CLASSES ARE ENUMERABLE AND INCREASING!!
     
      int32 cl;
      for (int32 i=0;i<VocabSize();i++) {
        cl=int2class_[i];
        if (class2minint_.count(cl)==0) class2minint_[cl]=i; // mapping class -> start int
        class2maxint_[cl]=i; // mapping class -> max int
      }

      ExpectMarker(in,binary,"<words>");
      std::string wrd;

      // read vocabulary
      for (int32 i=0;i<VocabSize();i++) {
        ReadMarker(in,binary,&wrd);
        word2int_[wrd]=i;
        int2word_[i]=wrd;
      }
  
      // prepare activation layers
      h_.setZero(HiddenSize());
      y_.setZero(VocabSize());
      cl_.setZero(ClassSize());       

      KALDI_LOG << "RNN model is loaded! ";
    };

    void SetLatticeSymbols(const fst::SymbolTable& symtab) {
        for (int32 i=0;i<int2class_.size();i++) {
          int32 j=word2int_[symtab.Find(i)];
          intlat2intrnn[i]=j; intrnn2intlat[j]=i;
        } 
    }


    void Read(const string& filename) {
      bool binary;
      Input in(filename,&binary);
      Read(in.Stream(),binary);
      in.Close();
    }

    RNN(const string& filename) {
      Read(filename);
    }

    virtual ~RNN() {
    }

    BaseFloat Propagate(int32 lastW,int32 w,VectorXr* hOut=NULL,VectorXr* hIn=NULL) {
      BaseFloat nllh=OOVPenalty(); // negative log likelihood of P(w|lastW,hidden)
      if (!hIn) hIn=&h_; // if no explicit hidden vector is passed use the RNN-state one!
      if (!hOut) hOut=&h_; // if no explicit hidden vector is passed use the RNN-state one!
      VectorXr h_ac; // temporary variables for helping optimization

      h_ac.noalias()=-U1_*(*hIn); // s(t)=-T*s(t-1)
      if (IsIV(lastW)) h_ac-=V1_.col(lastW); // s(t)=-s(t)-U*w-1(t)

      // activate hidden layer (sigmoid) and determine updated s(t)
      hOut->noalias()=VectorXr(VectorXr(h_ac.array().exp()+1.0).array().inverse());

      if (IsIV(w)){
        // evaluate classes: c(t)=W*s(t) + activation class layer (softmax)
        cl_.noalias()=VectorXr((W2_*(*hOut)).array().exp());
        // evaluate post. distribution for all words within that class: y(t)=V*s(t)
        int b=class2minint_[int2class_[w]];
        int n=class2maxint_[int2class_[w]]-b+1;

        // determine distribution of class of the predicted word
        // activate class part of the word layer (softmax)
        y_.segment(b,n).noalias()=VectorXr((V1_.middleRows(b,n)*(*hOut)).array().exp());
        nllh=-log(y_(w)*cl_(int2class_[w])/cl_.sum()/y_.segment(b,n).sum());
      }
      return nllh;
    }

  void TreeTraverse(const fst::SymbolTable& wordsym, CompactLattice& lat, fst::StdFst::StateId i,std::string sentence="",BaseFloat lms=0,BaseFloat ams=0) {
    if (lat.NumArcs(i)==0) {
      KALDI_LOG<<" "<<sentence<<" "<<lms<<" "<<ams;
      return;
    }
    for (fst::ArcIterator<CompactLattice> aiter(lat, i); !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      const CompactLatticeWeight &wgt = aiter.Value().weight;
      BaseFloat rnnscore=0;
      //TODO compute word posterior! 
      TreeTraverse(wordsym,lat,arc.nextstate,sentence+" "+wordsym.Find(arc.olabel),rnnscore+lms+wgt.Weight().Value1(),ams+wgt.Weight().Value2());
    }
  }

};


int main(int argc, char *argv[]) {
  try {
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Extracts N-best paths from lattices using given acoustic scale. \n"
        "Rescores them using an RNN model and given lm scale and writes it out as FST\n"
        "Usage: lattice-rnnrescore [options] dict lattice-rspecifier rnn-model lattice-wspecifier\n"
        " e.g.: lattice-rnnrescore --acoustic-scale=0.0625 --lm-scale=1.5 --iv-penalty=3 --oov-penalty=10 --n=10 WSJ.word-sym-tab ark:in.lats WSJ.rnn ark:nbest.lats\n";
      
    ParseOptions po(usage);
    BaseFloat lambda = 0.75;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat oov_penalty = 11; // assumes vocabularies around 60k words 
    BaseFloat iv_penalty = 0;
    std::string text_file = "";
    bool no_oovs = false;
    RNN::int32 n = 10;
    

    po.Register("lambda", &lambda, "Weighting factor between 0 and 1 for the given RNN LM" );
    po.Register("write-as-text", &text_file, "TODO Dump the nbests in a text formatted file");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("oov-penalty", &oov_penalty, "A reasonable value is ln(vocab_size)" );
    po.Register("iv-penalty", &iv_penalty, "TODO Can be used to tune performance" );
    po.Register("no-oovs", &no_oovs, "TODO Will cause rescoring to abort in cases of OOV words!" ); 
    po.Register("n", &n, "Number of distinct paths >= 1");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      KALDI_EXIT << "Wrong arguments!";
    }

    std::string wordsymtab_filename = po.GetArg(1),
        rnnmodel_filename = po.GetArg(3),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(4);

    // read the dictionary
    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(wordsymtab_filename)))
      KALDI_EXIT << "Could not read symbol table from file " << wordsymtab_filename;

    // Read as regular lattice-- this is the form we need it in for efficient
    // pruning.
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    // initialize our RNN
    RNN myRNN(rnnmodel_filename); 
    myRNN.SetLatticeSymbols(*word_syms);

    RNN::int32 n_done = 0; // there is no failure mode, barring a crash.
    RNN::int64 n_paths_out = 0;

    if (acoustic_scale == 0.0)
      KALDI_EXIT << "Do not use a zero acoustic scale (cannot be inverted)";
    if (lambda == 0.0)
      KALDI_EXIT << "Do not use lambda==0, it has no effect";

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();
      Lattice rlat;

      lattice_reader.FreeCurrent();

      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);\

      fst::Reverse(lat, &rlat);
      Lattice nbestr_lat, nbest_lat;
      fst::ShortestPath(rlat, &nbestr_lat, n);
      fst::Reverse(nbestr_lat, &nbest_lat);

      if (nbestr_lat.Start() != fst::kNoStateId)
        n_paths_out += nbestr_lat.NumArcs(nbestr_lat.Start());

      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &nbest_lat);

      fst::LifoQueue<fst::StdFst::StateId> q; 
      CompactLattice nbest_clat;

      ConvertLattice(nbest_lat, &nbest_clat);
      compact_lattice_writer.Write(key, nbest_clat);
      n_done++;

      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &nbest_clat);\

      myRNN.TreeTraverse(*word_syms,nbest_clat,nbest_clat.Start(),key);
    }

    KALDI_LOG << "Did N-best algorithm to " << n_done << " lattices with n = "
              << n << ", average actual #paths is "
              << (n_paths_out/(n_done+1.0e-20));
    KALDI_LOG << "Done " << n_done << " utterances.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
