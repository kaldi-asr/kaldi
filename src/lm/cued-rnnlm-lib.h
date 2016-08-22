#ifndef KALDI_LM_CUED_RNNLM_LIB_H_
#define KALDI_LM_CUED_RNNLM_LIB_H_

#include <string>
#include <vector>
#include "util/stl-utils.h"
#include "base/timer.h"
#include "matrix/kaldi-matrix.h"

#include <omp.h>
#include <time.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "math.h"

using namespace std;

namespace cued_rnnlm {
using kaldi::Timer;
using kaldi::Matrix;
using kaldi::MatrixBase;
using kaldi::SubMatrix;

#define     SUCCESS             0
#define     FILEREADERROR       1
#define     ARGSPARSEERROR      2
#define     EOL                 -2
#define     FILLEDNULL          1e8
#define     CHECKNUM            9999999
#define     MINRANDINITVALUE    -0.1
#define     MAXRANDINITVALUE    0.1
// #define     CONSTLOGNORM        9
#define     NOTSTOREMODEL
#define     NUM_THREAD          8
#define     MAX_WORD_LINE       4008
// #define     INPUTEMBEDDINGCPU
#define     FUDGE_FACTOR        1e-8
#define     MBUPDATEOUTPUTLAYER
// #define     RMSPROP
#define     RMSPROPCOFF         0.9
#define     LOGADD
typedef     map<string, int>    WORDMAP;
typedef     float               real;

typedef  short int          int16;
typedef unsigned short int  uint16;
typedef long int            int32;
typedef unsigned long int   ulint32;

#define  MAX_STRINGLEN       1000
// if two clusters includes the speakers in the same session, then never merge them.
#define NOTMERGESPEAKERSINSAMESESSION
#define MIN_BICVALUE  -1e10

#define MAX_STRING 1000
#define INVALID_INT 1e8
// #define CACHESIZE 1000
#define CUDA
// #define CUDADEBUG
// #define CPUSIDE   // This macro runs GPU and CPU in the mean time when GPU is set
// #define TIMEINFO
// #define TIMEINFO1
#define DUMPENTROPY
// #define VARYLEARNRATE  // apply different learnning rate in layer1 and layer0
// #define USEMOMENTUM
#define WEIGHTDECAY_MB  1   // the number of minibatch when the weight matrix decays
#define RESETVALUE 0.1
// typedef double real;
typedef float real;
// #define NOSENTSPLICE      // rnnlm training without sentence splice, just for speed comparison
#define isdouble(real)  (sizeof(real) == sizeof(double))
typedef double direct_t;


#define foreach_row(_i,_m)    for (size_t _i = 0; _i < (_m)->rows(); _i++)
#define foreach_column(_j,_m) for (size_t _j = 0; _j < (_m)->cols(); _j++)
#define foreach_coord(_i,_j,_m) for (size_t _j = 0; _j < (_m)->cols(); _j++) for (size_t _i = 0; _i < (_m)->rows(); _i++)


// #define     DEBUG

void printusage(char *str);

class arguments {
 protected:
  int argc;
  char **argv;
  map <string, string> argmap;
 public:
  arguments(int n, char **v): argc(n-1), argv(v+1) {
    int i;
    string str;
    i = 0;
    while (i < argc)
    {
      if (argv[i][0] == '-') {
        str = argv[i];
        if (i<argc-1 && argv[i+1][0] != '-')
        {
          argmap[str] = argv[i+1];
          i += 2;
        }
        else
        {
          argmap[str] = "true";
          i += 1;
        }
      }
    }
  }

  bool empty() {
    return (argc == 0);
  }

  string find (string str) {
    if (argmap.find(str) == argmap.end()) {
      return string ("EMPTY");
    }
    return argmap[str];
  }
};

bool isEmpty(string str);

int string2int (string str);

float string2float (string str);

void parseArray (string str, vector<int> &layersizes);

float randomv(float min, float max);

int getline (char *line, int &max_words_line, FILE *&fptr);

float logadd (float x, float y);

class FILEPTR
{
 protected:
  FILE *fptr;
  int i;
  string filename;
 public:
  FILEPTR() {
    fptr = NULL;
  }

  ~FILEPTR() {
    if (fptr) {
      fclose(fptr);
    }
    fptr = NULL;
  }

  void open (string fn) {
    filename = fn;
    fptr = fopen (filename.c_str(), "rt");
    if (fptr == NULL)
    {
      printf ("ERROR: Failed to open file: %s\n", filename.c_str());
      exit (0);
    }
  }

  void close() {
    if (fptr) {
      fclose(fptr);
      fptr = NULL;
    }
  }

  bool eof()
  {
    return feof(fptr);
  }

  int readint () {
    if (!feof(fptr)) {
      if(fscanf (fptr, "%d", &i) != 1) {
        if (!feof(fptr)) {
          printf ("Warning: failed to read feature index from text file (%s)\n",
                  filename.c_str());
        }
      }
      return i;
    }
    else {
      return INVALID_INT;
    }
  }

  void readline (vector<string> &linevec, int &cnt) {
    linevec.clear();
    char word[1024];
    char c;
    int index = 0;
    cnt = 0;
    while (!feof(fptr)) {
      c = fgetc(fptr);
      // getvalidchar (fptr, c);
      if (c == '\n') {
        if (cnt == 0 && strcmp(word, "<s>") != 0) {
          linevec.push_back("<s>");
          cnt ++;
        }
        if (index > 0)
        {
          word[index] = 0;
          linevec.push_back(word);
          cnt ++;
        }
        break;
      }
      else if ((c == ' ' || c == '\t') && index == 0) // space in the front of line
      {
        continue;
      }
      else if ((c == ' ' || c=='\t') && index > 0) // space in the middle of line
      {
        word[index] = 0;
        if (cnt == 0 && strcmp(word, "<s>") != 0) {
          linevec.push_back("<s>");
          cnt ++;
        }
        linevec.push_back(word);
        index = 0;
        cnt ++;
      }
      else {
        word[index] = c;
        index ++;
      }
    }
    if (cnt > 0 && strcmp(word, "</s>") != 0) {
      linevec.push_back("</s>");
      cnt ++;
    }
  }
};

class matrix
{
 private:
  Matrix<real>* host_data;
 public:
  matrix(): host_data(NULL) {}
  matrix(size_t nr, size_t nc) {
    if (nc == 1) {
      host_data = new Matrix<real>(nr, nc, kaldi::kSetZero, kaldi::kStrideEqualNumCols);
    }
    host_data = new Matrix<real>(nr, nc, kaldi::kSetZero, kaldi::kDefaultStride);
  }
  ~matrix () {
    freemem();
  }
  size_t Sizeof () {
    return (host_data->NumRows() * host_data->NumCols() * sizeof(real));
  }
  size_t nelem () {
    return (host_data->NumRows() * host_data->NumCols());
  }
  // asign value on CPU
  void assignhostvalue (size_t i, size_t j, real v) {
    (*host_data)(i, j) = v;
  }
  void addhostvalue (size_t i, size_t j, real v) {
    (*host_data)(i, j) += v;
  }
  real fetchhostvalue (size_t i, size_t j) {
    return (*host_data)(i, j);
  }

  size_t rows () {
    return host_data->NumRows();
  }
  size_t cols () {
    return host_data->NumCols();
  }
  void freemem () {
    if (host_data) {
      delete host_data;
    }
  }

  real& operator() (int i, int j) const {
    return (*host_data)(i, j);
  }

  const real& operator() (int i, int j) {
    return (*host_data)(i, j);
  }

  Matrix<real>* GetMatrixPointer() {
    return host_data;
  }

  real* gethostdataptr () {
    // only return the data pointer when it's continuously stored
    KALDI_ASSERT(host_data->NumCols() == 1);
    return host_data->Data();
  }

  real *gethostdataptr(int i, int j) {
    return &(*host_data)(i, j);
  }

  // initialize all element (both GPU and CPU) in matrx with v
  void initmatrix (int v = 0) {
    host_data->Set(v);
  }

  void hostrelu (float ratio)
  {
    KALDI_ASSERT(host_data->NumCols() == 1);

    host_data->Scale(ratio);
    host_data->ApplyFloor(0);
  }

  void hostsigmoid() {
    KALDI_ASSERT(host_data->NumCols() == 1);

    host_data->Sigmoid(*host_data);
  }

  void hostsoftmax() {
    KALDI_ASSERT(host_data->NumCols() == 1);
    host_data->ApplySoftMax();
  }

  void hostpartsoftmax(int swordid, int ewordid) {
    KALDI_ASSERT(host_data->NumCols() == 1);
    SubMatrix<real> t(*host_data, swordid, 1 + ewordid - swordid, 0, 1);
    t.ApplySoftMax();
  }

  void random(float min, float max) {
    int i, j;
    float v;
    for (i = 0; i < host_data->NumRows(); i++) {
      for (j = 0; j < host_data->NumCols(); j++) {
        v = randomv(min, max) + randomv(min,max) + randomv(min, max);
        (*host_data)(i, j) = v;
      }
    }
  }
};

class RNNLM
{
 protected:
  string inmodelfile, trainfile, validfile, nglmstfile,
         testfile, inputwlist, outputwlist, feafile;
  vector<int> layersizes;
  map<string, int> inputmap, outputmap;
  vector<string>  inputvec, outputvec, ooswordsvec;
  vector<float>   ooswordsprob;
  vector<matrix *> layers, neu_ac;
  matrix *layer0_hist, *neu0_ac_hist, *layer0_fea, *feamatrix,
         *neu0_ac_fea, *layerN_class, *neuN_ac_class, *lognorms;
  float logp, llogp, nwordspersec,
        lognorm_mean, lognorm_var,
        lognormconst, lambda, version, reluratio,
        lmscale, ip;
  int minibatch, debug, iter, traincritmode,
      inputlayersize, outputlayersize, num_layer, wordcn, trainwordcnt,
      validwordcnt, independent, inStartindex, inOOSindex,
      outEndindex, outOOSindex, bptt, bptt_delay, counter,
      fullvocsize, N, prevword, curword, num_oosword, nthread,
      num_fea, dim_fea, nclass, nodetype;
  int *host_curwords, *word2class, *classinfo,
      *host_curclass;
  bool binformat;
  float *resetAc; // allocate memory

 public:
  RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1,
        const vector<int> &lsizes, int fvocsize, bool bformat, int debuglevel);

  ~RNNLM();

  int getHiddenLayerSize() const {return layersizes[1];}
  void copyToHiddenLayer(const vector<float> &hidden);
  void fetchHiddenLayer(vector<float> *out);

  float computeConditionalLogprob(                                              
      int current_word, const std::vector<int> &history_words,                            
      const std::vector<float> &context_in, std::vector<float> *context_out);

  bool calppl (string testfilename, float lambda, string nglmfile);

  bool calnbest (string testfilename, float lambda, string nglmfile);

  void InitVariables ();
  void LoadRNNLM(string modelname);
  void LoadBinaryRNNLM(string modelname);
  void LoadTextRNNLM(string modelname);
  void ReadWordlist (string inputlist, string outputlist);
  void WriteWordlist (string inputlist, string outputlist);
  void printPPLInfo ();
  void printSampleInfo ();

  void init();

  void setLognormConst (float v)  {lognormconst = v;}
  void setNthread (int n)         {nthread = n;}
  void setLmscale (float v)       {lmscale = v;}
  void setIp (float v)            {ip = v;}
  void setFullVocsize (int n);
  void copyRecurrentAc ();
  void ResetRechist();
  float forward (int prevword, int curword);
  void ReadUnigramFile (string unigramfile);
  void matrixXvector (const MatrixBase<real> &src, const MatrixBase<real> &wgt,
                            MatrixBase<real> &dst, int nr, int nc);
  void allocMem (vector<int> &layersizes);

};


}  // namespace cued_rnnlm

#endif  // KALDI_LM_CUED_RNNLM_LIB_H_
