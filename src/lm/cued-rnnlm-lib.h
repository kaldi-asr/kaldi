#ifndef KALDI_LM_CUED_RNNLM_LIB_H_
#define KALDI_LM_CUED_RNNLM_LIB_H_

#include <string>
#include <vector>
#include "util/stl-utils.h"

#include <omp.h>
#include <time.h>
/*
#include "head.h"
#include "helper.h"
#include "fileops.h"
#include "DataType.h"
// */

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
        if (cnt == 0 && word[0] != '<') {
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
        if (cnt == 0 && word[0] != '<') {
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
    if (cnt > 0 && word[0] != '<') {
      linevec.push_back("</s>");
      cnt ++;
    }
  }
};


class auto_timer
{
  timespec time_start, time_end;
  real sec;
  real nsec;
  real acctime;
public:
  void start () {
    clock_gettime (CLOCK_REALTIME, &time_start);
  }

  void end() {
       clock_gettime (CLOCK_REALTIME, &time_end);
  }

  void add() {
    end();
    if (time_end.tv_nsec - time_start.tv_nsec < 0) {
      nsec = 1000000000 + time_end.tv_nsec - time_start.tv_nsec;
      sec = time_end.tv_sec - time_start.tv_sec - 1;
    }
    else {
      nsec = time_end.tv_nsec - time_start.tv_nsec;
      sec = time_end.tv_sec - time_start.tv_sec;
    }
    acctime += sec + nsec * 1.0 / 1000000000;
  }

  real stop() {
    end();
    if (time_end.tv_nsec - time_start.tv_nsec < 0) {
      nsec = 1000000000 + time_end.tv_nsec - time_start.tv_nsec;
      sec = time_end.tv_sec - time_start.tv_sec - 1;
    }
    else {
      nsec = time_end.tv_nsec - time_start.tv_nsec;
      sec = time_end.tv_sec - time_start.tv_sec;
    }
    return sec + nsec * 1.0/1000000000;
  }

  real getacctime () {
    return acctime;
  }

  void clear() {
    sec = 0.0;
    nsec = 0.0;
    acctime = 0.0;
  }

  auto_timer () {
    sec = 0.0;
    nsec = 0.0;
    acctime = 0.0;
    time_start.tv_sec = 0;
    time_end.tv_sec = 0;
    time_start.tv_nsec = 0;
    time_end.tv_nsec = 0;
  }
};

class matrix
{
 private:
  real* host_data;
  size_t nrows;
  size_t ncols;
  size_t size;
 public:
  matrix (): host_data(NULL), nrows(0), ncols(0) {}
  matrix (size_t nr, size_t nc) {
    nrows = nr;
    ncols = nc;
    size = sizeof(real) * ncols * nrows;
    host_data = (real *) malloc (size);
  }
  ~matrix () {
    if (host_data) {
      free (host_data);
      host_data = NULL;
    }
  }
  size_t Sizeof () {
    return (nrows * ncols * sizeof(real));
  }
  size_t nelem () {
    return (nrows * ncols);
  }
  // asign value on CPU
  void assignhostvalue (size_t i, size_t j, real v) {
    host_data[i + j * nrows] = v;
  }
  void addhostvalue (size_t i, size_t j, real v) {
    host_data[i + j * nrows] += v;
  }
  real fetchhostvalue (size_t i, size_t j) {
    return host_data[i + j * nrows];
  }

  void setnrows (size_t nr) {
    nrows = nr;
  }
  void setncols (size_t nc) {
    ncols = nc;
  }
  size_t rows () {
    return nrows;
  }
  size_t cols () {
    return ncols;
  }
  void freemem () {
    free (host_data);
    ncols = 0;
    nrows = 0;
    size = 0;
  }

  real& operator() (int i, int j) const {
    assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
    return host_data[i + j * nrows];
  }

  const real& operator() (int i, int j) {
    assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
    return host_data[i + j * nrows];
  }
  real* gethostdataptr () {
    return host_data;
  }

  real *gethostdataptr(int i, int j) {
    return &host_data[i+j*nrows];
  }

  // initialize all element (both GPU and CPU) in matrx with v
  void initmatrix (int v = 0) {
    memset (host_data, v, Sizeof());
  }

  void hostrelu (float ratio)
  {
    assert (ncols == 1);
    for (int i = 0; i < nrows; i++) {
      if (host_data[i] > 0) {
        host_data[i] *= ratio;
      }
      else {
        host_data[i] = 0;
      }
    }
  }

  void hostsigmoid() {
    assert (ncols == 1);
    for (int i = 0; i< nrows; i++) {
      host_data[i] = 1.0 / (1 + exp(-host_data[i]));
    }
  }

  void hostsoftmax() {
//        int a, maxi;
    int a;
    float v, norm, maxv = 1e-8;
    assert (ncols == 1);
    maxv = 1e-10;
    for (a = 0; a < nrows; a++) {
      v = host_data[a];
      if (v > maxv) {
        maxv = v;
//                maxi = a;
      }
    }
    norm = 0;

    for (a = 0; a < nrows; a++) {
      v = host_data[a] - maxv;
      host_data[a] = exp(v);
      norm += host_data[a];
    }
    for (a = 0; a < nrows; a++) {
      v = host_data[a] / norm;
      host_data[a] = v;
    }
  }

  void hostpartsoftmax(int swordid, int ewordid) {
//        int a, maxi;
    int a;
    float v, norm, maxv = 1e-8;
    assert (ncols == 1);
    maxv = 1e-10;
    for (a = swordid; a <= ewordid; a++) {
      v = host_data[a];
      if (v > maxv) {
        maxv = v;
//                maxi = a;
      }
    }
    norm = 0;

    for (a = swordid; a <= ewordid; a++) {
      v = host_data[a] - maxv;
      host_data[a] = exp(v);
      norm += host_data[a];
    }
    for (a = swordid; a <= ewordid; a++) {
      v = host_data[a] / norm;
      host_data[a] = v;
    }
  }

  void random(float min, float max) {
    int i, j;
    float v;
    for (i = 0; i < nrows; i++) {
      for (j = 0; j < ncols; j++) {
        v = randomv(min, max) + randomv(min,max) + randomv(min, max);
        host_data[i + j * nrows] = v;
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
  auto_timer timer_sampler, timer_forward, timer_output, timer_backprop, timer_hidden;

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
  void matrixXvector (float *ac0, float *wgt1, float *ac1, int nrow, int ncol);
  void allocMem (vector<int> &layersizes);

  // functions for using additional feature file in input layer
  void ReadFeaFile(string filestr);
  void setFeafile (string str)  {feafile = str;}

};


}  // namespace cued_rnnlm

#endif  // KALDI_LM_CUED_RNNLM_LIB_H_
