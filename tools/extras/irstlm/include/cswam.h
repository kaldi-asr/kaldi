/******************************************************************************
 IrstLM: IRST Language Model Toolkit, compile LM
 Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy
 
 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 
 ******************************************************************************/
#ifndef MF_CSWAM_H
#define MF_CSWAM_H

#ifdef HAVE_CXX0
#include <unordered_map>
#else
#include <map>
#endif

#include <vector>

namespace irstlm {

typedef struct{
    
    float*  M;  //mean vectors
    float*  S;  //variance vectors
    //training support items
    float   eC; //support set size
    float   mS; //mean variance
    
} Gaussian;

typedef struct{
    int      n;  //number of Gaussians
    float    *W; //weight vector
    Gaussian *G; //Gaussians
} TransModel;

typedef struct{
    int      word;      //word code
    float    score;  //score (mutual information)
} Friend;

typedef std::vector<Friend> FriendList; //list of word Friends
#ifdef HAVE_CXX0
typedef std::unordered_map<int,float> src_map; //target to source associative memory
#else
typedef std::map<int,float> src_map; //target to source associative memory
#endif

class cswam {
    
    //data
    dictionary* srcdict; //source dictionary
    dictionary* trgdict; //target dictionary
    doc* srcdata;   //source training data
    doc* trgdata;   //target trainign data
    FriendList* friends; //prior list of translation candidates
    
    //word2vec
    float     **W2V;   //vector for each source word
    int       D;       //dimension of vector space
    
    
    //model
    TransModel *TM;
    float DistMean,DistVar; //distortion mean and variance
    float DistA,DistB;      //gamma parameters
    float NullProb; //null probability
    
    //settings
    bool normalize_vectors;
    bool train_variances;
    double fix_null_prob;
    bool use_null_word;
    bool verbosity;
    float min_variance;
    int distortion_window;
    bool distortion_mean;
    bool distortion_var;
    bool use_beta_distortion;
    int minfreq;
    bool incremental_train;
    
    //private info shared among threads
    int trgBoD;        //code of segment begin in target dict
    int trgEoD;        //code of segment end in target dict
    int srcBoD;        //code of segment begin in src dict
    int srcEoD;        //code of segment end in src dict
    
    float ****A;       //expected counts
    float **Den;       //alignment probs
    float *localLL;    //local log-likelihood
    int **alignments;  //word alignment info
    int threads;       //number of threads
    int bucket;        //size of bucket
    int iter;          //current iteration
    int M1iter;        //iterations with model 1
    
    //Model 1 initialization private variables
   
    src_map* prob;  //model one probabilities
    src_map** loc_efcounts;  //expected count probabilities
    float **loc_ecounts;     //expected count probabilities
    src_map* efcounts;  //expected count probabilities
    float *ecounts;     //expected count probabilities
    
    struct task {      //basic task info to run task
        void *ctx;
        void *argv;
    };
    
    
public:
    
    cswam(char* srcdatafile,char* trgdatafile, char* word2vecfile,
          bool forcemodel,
          bool usenull,double fix_null_prob,
          bool normv2w,
          int model1iter,
          bool trainvar,float minvar,
          int distwin,bool distbeta, bool distmean,bool distvar,
          bool verbose);
    
    ~cswam();
    
    void loadword2vec(char* fname);
    void randword2vec(const char* word,float* vec,int it=0);
    void initModel(char* fname);
    void initEntry(int entry);
    int saveModel(char* fname);
    int saveModelTxt(char* fname);
    int loadModel(char* fname,bool expand=false);
    
    void initAlphaDen();
    void freeAlphaDen();
    
    
    float LogGauss(const int dim,const float* x,const float *m, const float *s);
    
    float LogDistortion(float d);
    float LogBeta(float x,  float a,  float b);
    void EstimateBeta(float &a, float &b,  float m,  float s);
    
    float Delta( int i, int j, int l=1, int m=1);
    
    void expected_counts(void *argv);
    static void *expected_counts_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->expected_counts(t.argv);return NULL;
    };
    
    void maximization(void *argv);
    static void *maximization_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->maximization(t.argv);return NULL;
    };
    
    void expansion(void *argv);
    static void *expansion_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->expansion(t.argv);return NULL;
    };
    
    void contraction(void *argv);
    static void *contraction_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->contraction(t.argv);return NULL;
    };
    
    
    void M1_ecounts(void *argv);
    static void *M1_ecounts_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->M1_ecounts(t.argv);return NULL;
    }
   
    void M1_collect(void *argv);
    static void *M1_collect_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->M1_collect(t.argv);return NULL;
    }
    
    void M1_update(void *argv);
    static void *M1_update_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->M1_update(t.argv);return NULL;
    }
    
    void M1_clearcounts(bool clearmem=false);
        
    void findfriends(FriendList* friends);
    
    
    
    int train(char *srctrainfile,char *trgtrainfile,char* modelfile, int maxiter,int threads=1);
    
    void aligner(void *argv);
    static void *aligner_helper(void *argv){
        task t=*(task *)argv;
        ((cswam *)t.ctx)->aligner(t.argv);return NULL;
    };
    
    
    int test(char *srctestfile, char* trgtestfile, char* modelfile,char* alignmentfile, int threads=1);
    
};
} //namespace irstlm
#endif
