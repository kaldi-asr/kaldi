// latbin/lattice-nbest.cc

// Copyright 2009-2011  Microsoft Corporation

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


    void Read(const string& filename) {
      bool binary;
      Input in(filename,&binary);
      Read(in.Stream(),binary);
      in.Close();
    }

    RNN(const string& filename) {
      Read(filename);
    };

    virtual ~RNN() {
    };

    // propagate a sequence through the RNN
    // seq - the sequence of words w0 w1 w2 ... in RNN int32 code, at least two words. First word w0 is the history word!
    // prob - vectors with posterior probabilities for w1 w2 ... - NB: seq.count()-1 == prob.count() 
    // hidden - pointer to a hidden state vector which is used for initialization. After processing will contain the updated hidden state
    //        - if omitted, will use the member state vector h_ instead !!
    void PropagateSeq(const vector<int32>& seq, vector<realn>* prob, VectorXr* hidden=NULL) {
      VectorXr h_ac; // temporary variables for helping optimization
      int32 seqlen=seq.size()-1;
      if (seqlen<1) return; // at least two words must be passed!

      MatrixXr H(seqlen,HiddenSize()); // matrix to store hidden state activations

      if (!hidden) hidden=&h_; // if no explicit hidden vector is passed use the RNN-state one!

      int32 w;
      for (int32 i=0;i<seqlen;i++) {
        w=seq[i];
        h_ac.noalias()=-U1_*(*hidden); // h(t)=-U1*h(t-1)
        if (IsIV(w)) h_ac-=V1_.col(w); // h(t)=-h(t)-V1*w-1(t)

        // activate hidden layers (sigmoid) and determine updated s(t)
        (*hidden).noalias()=VectorXr(VectorXr(h_ac.array().exp()+1.0).array().inverse());
        H.row(i)=(*hidden);
      }
      // evaluate classes: c(t)=Cl*h(t) + activation class layer (softmax)
      MatrixXr C=MatrixXr((H*Cl_.transpose()).array().exp());
      VectorXr c_norm(seqlen); for (int32 i=0;i<seqlen;i++) c_norm(i)=C.row(i).sum();
      realn y_norm;

      prob->clear();
      for (int32 i=0;i<seqlen;i++) {
      w=seq[i+1]; // predicted word 
      if (IsIV(w)) {
        // evaluate post. distribution for all words within that class: y(t)=V*s(t)
        int32 b=class2minint_[int2class_[w]];
        int32 n=class2maxint_[int2class_[w]]-b+1;

        // determine distribution of class of the predicted word
        // activate class part of the word layer (softmax)
        y_.segment(b,n).noalias()=VectorXr((W2_.middleRows(b,n)*VectorXr(H.row(i))).array().exp());
        y_norm=y_.segment(b,n).sum();
        //std::cout << y_norm << " "<<c_norm(i)<<std::endl;
        prob->push_back((y_(w)*C.row(i))(int2class_[w])/c_norm(i)/y_norm);
      } else prob->push_back(0);
    }
    KALDI_ASSERT(seq.size()-1==prob->size());
  }

  void Score(const vector<int32>& seq, float oov_penalty, Vector<realn>* score) {
    KALDI_ASSERT(seq.size()>=2);

    vector<realn> score_part;
    vector<int32> seq_part;
    score->Resize(seq.size(),kUndefined);

    bool firstRun=true;
    VectorXr hOld=h_,hNew;

    int32 i=0,ix=0;
    while (ix<seq.size()) {
      seq_part.clear();seq_part.push_back(0);seq_part.push_back(0);
      while (seq[i]!=0) {seq_part.push_back(seq[i]); i++;}
      seq_part.push_back(0);i++;
      h_=hOld;
      PropagateSeq(seq_part,&score_part);
      if (firstRun) {
        firstRun=false;
        hNew=h_;
      }
      double p,sumllh=0;
      for (int32 s=1;s<score_part.size();s++) {
        p=score_part[s]==0?(oov_penalty):(score_part[s]);
        sumllh+=log10(p);
        (*score)(ix)=p;
        ix++;
      }
      std::cout<<sumllh<<"\n";
    }
    h_=hNew;
  }

};

int main(int argc, char *argv[]) {
  try {
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Rescore N-best paths in lattices with an RNN model and write it out as FST\n"
        "Usage: lattice-rnnrescore [options] dict rnn-model lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-rnnrescore --acoustic-scale=0.0625 --iv-penalty=3 --oov-penalty=10 --lambda=0.5 --n=10 WSJ.word-sym-tab WSJ.rnn ark:1.lats ark:nbest.lats\n";
      
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lambda = 0.75;
    BaseFloat oov_penalty = 11; // assumes vocabularies around 60k words 
    BaseFloat iv_penalty = 0;
    bool no_oovs = false;
    RNN::int32 n = 10;
    

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("lambda", &lambda, "Weight for the RNN model, between 0 and 1");
    po.Register("oov-penalty", &oov_penalty, "A reasonable value is ln(vocab_size)" );
    po.Register("iv-penalty", &iv_penalty, "Can be used to tune performance" );
    po.Register("no-oovs", &no_oovs, "Will cause rescoring to abort in cases of OOV words!" ); 
    po.Register("n", &n, "Number of distinct paths >= 1");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      KALDI_EXIT << "Wrong arguments!";
    }

    std::string wordsymtab_filename = po.GetArg(1),
        rnnmodel_filename = po.GetArg(2),
        lats_rspecifier = po.GetArg(3),
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

    RNN::int32 n_done = 0; // there is no failure mode, barring a crash.
    RNN::int64 n_paths_out = 0;

    if (acoustic_scale == 0.0)
      KALDI_EXIT << "Do not use a zero acoustic scale (cannot be inverted)";
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

      CompactLattice nbest_clat;
      ConvertLattice(nbest_lat, &nbest_clat);
      compact_lattice_writer.Write(key, nbest_clat);
      n_done++;
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
