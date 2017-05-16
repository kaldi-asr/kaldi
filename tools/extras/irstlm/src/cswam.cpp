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
 
 *******************************************************************************/

#include <sys/mman.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <pthread.h>
#include "thpool.h"
#include "crc.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "n_gram.h"
#include "util.h"
#include "dictionary.h"
#include "ngramtable.h"
#include "doc.h"
#include <algorithm>
#include <vector>
#include "cswam.h"

using namespace std;

namespace irstlm {

cswam::cswam(char* sdfile,char *tdfile, char* w2vfile,
             bool forcemodel,
             bool usenull,double fixnullprob,
             bool normvect,
             int model1iter,
             bool trainvar,float minvar,
             int distwin,bool distbeta,bool distmean,bool distvar,
             bool verbose){
    
    //actual model structure
    
    TM=NULL;
    A=NULL;
    Den=NULL;
    friends=NULL;
    efcounts=NULL;
    ecounts=NULL;
    loc_efcounts=NULL;
    loc_ecounts=NULL;
    
    //setting
    incremental_train=forcemodel;
    normalize_vectors=normvect;
    train_variances=trainvar;
    use_null_word=usenull;
    min_variance=minvar;
    distortion_window=distwin;
    distortion_mean=distmean;
    distortion_var=distvar;
    use_beta_distortion=distbeta;
    fix_null_prob=fixnullprob;
    DistMean=DistVar=0;  //distortion mean and variance
    DistA=DistB=0;    //beta parameters
    NullProb=0;
    M1iter=model1iter;
    
    //set mininum word frequency to collect friends
    minfreq=10;
    
    
    cout << "cswam configuration.\n";
    cout << "Vectors:  normalize [" << normalize_vectors << "] \n";
    cout << "Gaussian Variances: train [" << train_variances << "] min [" << min_variance << "] initial [" << min_variance * SSEED << "]\n";
    cout << "Null word: active [" << use_null_word << "] fix_null_prob [" << fix_null_prob << "]\n";
    cout << "Distortion model: window [" << distortion_window << "] use beta [" << use_beta_distortion << "] update mean [" << distortion_mean << "] update variance [" << distortion_var << "]\n";
    
    
    srandom(100); //ensure repicable generation of random numbers
    bucket=BUCKET;
    threads=1;
    verbosity=verbose;
    
    //create dictionaries
    srcdict=new dictionary(NULL,100000); srcdict->generate(sdfile,true);
    trgdict=new dictionary(NULL,100000); trgdict->generate(tdfile,true);
    
    //make aware of oov word
    srcdict->encode(srcdict->OOV());
    trgdict->encode(trgdict->OOV());
    
    trgBoD = trgdict->encode(trgdict->BoD());  //codes for begin/end sentence markers
    trgEoD = trgdict->encode(trgdict->EoD());
    
    srcBoD = srcdict->encode(srcdict->BoD());  //codes for begin/end sentence markers
    srcEoD = srcdict->encode(srcdict->EoD());

    
    //load word2vec dictionary
    W2V=NULL; D=0;
    loadword2vec(w2vfile);

    //check consistency of word2vec with target vocabulary
    
    
}

cswam::~cswam() {
    
    assert(A==NULL);
    
    if (TM){
        cerr << "Releasing memory of Translation Model\n";
        for (int e=0;e<trgdict->size();e++){
            for (int n=0;n<TM[e].n;n++){
                delete TM[e].G[n].M;delete TM[e].G[n].S;
            }
            delete [] TM[e].G; delete [] TM[e].W;
        }
        delete [] TM;
    }
    if (W2V){
        cerr << "Releasing memory of W2W\n";
        for (int f=0;f<srcdict->size();f++)
            if (W2V[f]!=NULL) delete [] W2V[f];
        delete [] W2V;
    }
    
    if (friends) delete [] friends;
    
    cerr << "Releasing memory of srcdict\n";
    delete srcdict;
    cerr << "Releasing memory of srcdict\n";
    delete trgdict;
    
    
}

void cswam::randword2vec(const char* word,float* vec,int it){
    
    //initialize random generator
    srandom(crc16_ccitt(word,strlen(word))+it);
    
    //generate random numbers between -1 and +1,
    //then scale and shift according to w2v
    for (int d=0;d<D;d++)
          vec[d]=(float)(MY_RAND * SSEED * min_variance);
}


void cswam::loadword2vec(char* fname){
    
    cerr << "Loading word2vec file " << fname;
    mfstream inp(fname,ios::in);
    
    long long w2vsize;
    inp >> w2vsize; cerr << " size= " << w2vsize;
    inp >> D ; cout << " dim= " << D  << "\n";
    
    assert(D>0 && D<1000);
    
    int srcoov=srcdict->oovcode();
    
    W2V=new float* [srcdict->size()];
    for (int f=0;f<srcdict->size();f++) W2V[f]=NULL;
    
    char word[100]; float dummy; int f;
    
    for (long long i=0;i<w2vsize;i++){
        inp >> word;
        f=srcdict->encode(word);
        if (f != srcoov){
            W2V[f]=new float[D];
            for (int d=0;d<D;d++) inp >> W2V[f][d];
        }
        else //skip this word
            for (int d=0;d<D;d++) inp >> dummy;
        
        if (!(i % 10000)) cerr<< ".";
    }
    cerr << "\n";
    
    
    cerr << "looking for missing source words in w2v\n";
    int newwords=0;
    for ( f=0;f<srcdict->size();f++){
        if (W2V[f]==NULL && f!=srcBoD && f!=srcEoD) {
            if (verbosity)
                cerr << "Missing src word in w2v: [" << f << "] " << srcdict->decode(f) << "\n";
            
            W2V[f]=new float[D];
            
            //generate random vectors with same distribution
            randword2vec(srcdict->decode(f),W2V[f]);
            
            newwords++;
            
            if (verbosity){
                for (int d=0;d<D;d++) cerr << " " << W2V[f][d]; cerr << "\n";}
            
        }
    }
    
    cerr << "Generated " << newwords << " missing vectors\n";
    
    
    if (normalize_vectors){
       cerr << "Normalizing vectors\n";
        for (f=0;f<srcdict->size();f++)
        if (W2V[f]!=NULL){
            float norm=0;
            for (int d=0;d<D;d++) norm+=W2V[f][d]*W2V[f][d];
            norm=sqrt(norm);
            for (int d=0;d<D;d++) W2V[f][d]/=norm;
        }
    }
    
};

void cswam::initEntry(int e){
    assert(TM[e].G==NULL);
    
    //allocate a suitable number of gaussians
    
    TM[e].n=(friends && friends[e].size()?friends[e].size():1);
  
    assert(TM[e].n>0);
    TM[e].G=new Gaussian [TM[e].n];TM[e].W=new float[TM[e].n];
    for (int n=0;n<TM[e].n;n++){
        
        TM[e].G[n].M=new float [D];
        TM[e].G[n].S=new float [D];
        
        TM[e].G[n].eC=0;
        TM[e].G[n].mS=0;
        
        TM[e].W[n]=1.0/(float)TM[e].n;
        
        if (friends && friends[e].size()){
            int f=friends[e][n].word; //initialize with source vector
            memcpy(TM[e].G[n].M,W2V[f],sizeof(float) * D);
        }
        else{
          randword2vec(trgdict->decode(e),TM[e].G[n].M,n);
        }
       
        for (int d=0;d<D;d++)
            TM[e].G[n].S[d]=min_variance * SSEED; //take a wide standard deviation
        
    }
    
}



//void oldinitEntry(int e){
//
//    assert(TM[e].G==NULL);
//    
//    //allocate a suitable number of gaussians
//    TM[e].n=(int)ceil(log((double)trgdict->freq(e)+1.1));
//    
//    //some exceptions if
//    
//    assert(TM[e].n>0);
//    TM[e].G=new Gaussian [TM[e].n];TM[e].W=new float[TM[e].n];
//    for (int n=0;n<TM[e].n;n++){
//        
//        TM[e].G[n].M=new float [D];
//        TM[e].G[n].S=new float [D];
//        
//        TM[e].G[n].eC=0;
//        TM[e].G[n].mS=0;
//        
//        TM[e].W[n]=1.0/(float)TM[e].n;
//        
//        //initialize with w2v value if the same word is also in src
//        int f=srcdict->encode(trgdict->decode(e));
//        float srcfreq=srcdict->freq(f);float trgfreq=trgdict->freq(e);
//        if (f!=srcdict->oovcode() && srcfreq/trgfreq < 1.1 && srcfreq/trgfreq > 0.9 && srcfreq < 10 &&  f!=srcBoD && f!=srcEoD){
//            memcpy(TM[e].G[n].M,W2V[f],sizeof(float) * D);
//            for (int d=0;d<D;d++) TM[e].G[n].S[d]=min_variance; //dangerous!!!!
//            if (verbosity)  cerr << "Biasing verbatim translation of " << srcdict->decode(f) << "\n";
//        }else{
//            //pick candidates from friends
//            
//            
//                randword2vec(trgdict->decode(e),TM[e].G[n].M,n);
//            
//            for (int d=0;d<D;d++)
//                TM[e].G[n].S[d]=W2Vsd[d] * 10; //take a wide standard deviation
//        }
//    }
//    
//}

void cswam::initModel(char* modelfile){
    
    //test if model is readable
    bool model_available=false;
    FILE* f;if ((f=fopen(modelfile,"r"))!=NULL){fclose(f);model_available=true;}
    
    if (model_available)
        loadModel(modelfile,true); //we are in training mode!
    else{
        cerr << "Initialize model\n";
        
        if (use_beta_distortion){
            DistMean=0.5;DistVar=1.0/12.0; //uniform distribution on 0,1
            EstimateBeta(DistA,DistB,DistMean,DistVar);
        }else{
            DistMean=0;DistVar=10; //gaussian distribution over -1,+1: almost uniform
        }
        
        TM=new TransModel[trgdict->size()];
        
        friends=new FriendList[trgdict->size()];
        findfriends(friends);
        
        for (int e=0; e<trgdict->size(); e++) initEntry(e);
        
    }
    //this can overwrite existing model
    if (use_null_word)
        NullProb=(fix_null_prob?fix_null_prob:0.05); //null word alignment probability
    
}

int cswam::saveModelTxt(char* fname){
    cerr << "Writing model into: " << fname << "\n";
    mfstream out(fname,ios::out);
    out << "=dist= " << DistMean << " " << DistVar << "\n";
    out << "=nullprob= " << NullProb << "\n";
    for (int e=0; e<trgdict->size(); e++){
        out << "=h= " << trgdict->decode(e) << " sz= " << TM[e].n <<  "\n";
        for (int n=0;n<TM[e].n;n++){
            out << "=w= " << trgdict->decode(e) << " w= " << TM[e].W[n] << " eC= " << TM[e].G[n].eC << " mS= " << TM[e].G[n].mS << "\n";
            out << "=m= " << trgdict->decode(e); for (int d=0;d<D;d++) out << " " << TM[e].G[n].M[d] ;out << "\n";
            out << "=s= " << trgdict->decode(e); for (int d=0;d<D;d++) out << " " << TM[e].G[n].S[d]; out << "\n";
        }
    }
    return 1;
}

int cswam::saveModel(char* fname){
    cerr << "Saving model into: " << fname << " ...";
    mfstream out(fname,ios::out);
    out << "CSWAM " << D << "\n";
    trgdict->save(out);
    out.write((const char*)&DistMean,sizeof(float));
    out.write((const char*)&DistVar,sizeof(float));
    out.write((const char*)&NullProb,sizeof(float));
    for (int e=0; e<trgdict->size(); e++){
        out.write((const char*)&TM[e].n,sizeof(int));
        out.write((const char*)TM[e].W,TM[e].n * sizeof(float));
        for (int n=0;n<TM[e].n;n++){
            out.write((const char*)TM[e].G[n].M,sizeof(float) * D);
            out.write((const char*)TM[e].G[n].S,sizeof(float) * D);
        }
    }
    out.close();
    cerr << "\n";
    return 1;
}

int cswam::loadModel(char* fname,bool expand){

    cerr << "Loading model from: " << fname << "...";
    mfstream inp(fname,ios::in);
    char header[100];
    inp.getline(header,100);
    cerr << header ;
    int r;
    sscanf(header,"CSWAM %d\n",&r);
    if (D>0 && r != D)
        exit_error(IRSTLM_ERROR_DATA, "incompatible dimension in model");
    else
        D=r;

    if (verbosity) cerr << "\nLoading dictionary ... ";
    dictionary* dict=new dictionary(NULL,1000000);
    dict->load(inp);
    dict->encode(dict->OOV());
    int current_size=dict->size();
    
    //expand the model for training or keep the model fixed for testing
    if (expand){
        if (verbosity)
            cerr << "\nExpanding model to include targer dictionary";
        dict->incflag(1);
        for (int code=0;code<trgdict->size();code++)
            dict->encode(trgdict->decode(code));
        dict->incflag(2);
    }
    //replace the trgdict with the model dictionary
    delete trgdict;trgdict=dict;
    trgdict->encode(trgdict->OOV());           //updated dictionary codes
    trgBoD = trgdict->encode(trgdict->BoD());  //codes for begin/end sentence markers
    trgEoD = trgdict->encode(trgdict->EoD());
    
    
    TM=new TransModel [trgdict->size()];
    
    if (verbosity) cerr << "\nReading parameters .... ";
    inp.read((char*)&DistMean, sizeof(float));
    inp.read((char*)&DistVar, sizeof(float));
    inp.read((char*)&NullProb,sizeof(float));
    
    cerr << "DistMean: " << DistMean << " DistVar: " << DistVar << " NullProb: " << NullProb << "\n";
    if (use_beta_distortion)
        EstimateBeta(DistA,DistB,DistMean,DistVar);
    
    for (int e=0; e<current_size; e++){
        inp.read((char *)&TM[e].n,sizeof(int));
        TM[e].W=new float[TM[e].n];
        inp.read((char *)TM[e].W,sizeof(float) * TM[e].n);
        TM[e].G=new Gaussian[TM[e].n];
        for (int n=0;n<TM[e].n;n++){
            TM[e].G[n].M=new float [D];
            TM[e].G[n].S=new float [D];
            inp.read((char *)TM[e].G[n].M,sizeof(float) * D);
            inp.read((char *)TM[e].G[n].S,sizeof(float) * D);
            TM[e].G[n].eC=0;TM[e].G[n].mS=0;
        }
    }
    inp.close();
    
    cerr << "\nInitializing " << trgdict->size()-current_size << " new entries .... ";
    for (int e=current_size; e<trgdict->size(); e++) initEntry(e);
    cerr << "\nDone\n";
    return 1;
}

void cswam::initAlphaDen(){
    
    //install Alpha[s][i][j] to collect counts
    //allocate if empty
    
    if (A==NULL){
        assert(trgdata->numdoc()==srcdata->numdoc());
        A=new float ***[trgdata->numdoc()];
        for (int s=0;s<trgdata->numdoc();s++){
            A[s]=new float **[trgdata->doclen(s)];
            for (int i=0;i<trgdata->doclen(s);i++){
                A[s][i]=new float *[TM[trgdata->docword(s,i)].n];
                for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                    A[s][i][n]=new float [srcdata->doclen(s)];
            }
        }
    }
    //initialize
    for (int s=0;s<trgdata->numdoc();s++)
        for (int i=0;i<trgdata->doclen(s);i++)
            for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                memset(A[s][i][n],0,sizeof(float) * srcdata->doclen(s));
    
    //allocate
    if (Den==NULL){
        Den=new float*[trgdict->size()];
        for (int e=0;e<trgdict->size();e++)
            Den[e]=new float[TM[e].n];
    }
        
    //initialize
    for (int e=0;e<trgdict->size();e++)
        memset(Den[e],0,sizeof(float)*TM[e].n);
}

void cswam::freeAlphaDen(){
    
    if (A!=NULL){
        for (int s=0;s<trgdata->numdoc();s++){
            for (int i=0;i<trgdata->doclen(s);i++){
                for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                    delete [] A[s][i][n];
                delete [] A[s][i];
            }
            delete [] A[s];
        }
        delete [] A;
        A=NULL;
    }
    
    if (Den!=NULL){
        for (int e=0;e<trgdict->size();e++) delete [] Den[e];
        delete [] Den;  Den=NULL;
    }
    
}

///*****
//pthread_mutex_t cswam_mut1;
//pthread_mutex_t cswam_mut2;
double cswam_LL=0; //Log likelihood

float logsum(float a,float b){
    if (b<a) return a + logf(1 + expf(b-a));
    else return b + logf(1+ expf(a-b));
}

int global_i=0;
int global_j=0;

float cswam::LogGauss(const int dim,const float* x,const float *m, const float *s){
    
    static float log2pi=1.83787; //log(2 pi)
    float dist=0; float norm=0;

    for (int i=0;i<dim;i++){
        assert(s[i]>0);
        dist+=(x[i]-m[i])*(x[i]-m[i])/(s[i]);
        norm+=s[i];
    }
    
    return -0.5 * (dist + dim * log2pi + logf(norm));
    
}


float cswam::LogBeta( float x,float a,float b){
    
    assert(x>0 && x <1);
    
    //disregard constant factor!
    
    return (a-1) * log(x) + (b-1) * log(1-x);
    
}


float cswam::Delta(int i,int j,int l,int m){
    
    i-=(use_null_word?1:0);
    l-=(use_null_word?1:0);
    
    float d=((i - j)>0?(float)(i-j)/l:(float)(i-j)/m);   //range is [-1,+1];
    if (use_beta_distortion) d=(d+1)/2;  //move in range [0,1];
    
     //reduce length penalty for short sentences
     if (l<=6 || m<=6) d/=2;
    
    return d;
}

float cswam::LogDistortion(float d){
    
    if (use_beta_distortion)
        return LogBeta(d,DistA,DistB);
    else
        return LogGauss(1,&d,&DistMean,&DistVar);
    
}



void cswam::expected_counts(void *argv){
    
    long long s=(long long) argv;
    
    ShowProgress(s, srcdata->numdoc());
    
    int trglen=trgdata->doclen(s); // length of target sentence
    int srclen=srcdata->doclen(s); //length of source sentence
    
    float den;float delta=0; //distortion
    
    //reset likelihood
    localLL[s]=0;
    
    //compute denominator for each source-target pair
    for (int j=0;j<srclen;j++){
        //qcout << "j: " << srcdict->decode(srcdata->docword(s,j)) << "\n";
        den=0;
        for (int i=0;i<trglen;i++)
            if ((use_null_word && i==0) || abs(i-j-1) <= distortion_window){
                delta=Delta(i,j,trglen,srclen);
                for (int n=0;n<TM[trgdata->docword(s,i)].n;n++){
                    if (!(TM[trgdata->docword(s,i)].W[n]>0))
                        cerr << trgdict->decode(trgdata->docword(s,i)) << " n:" << n << "\n";
                    assert(TM[trgdata->docword(s,i)].W[n]>0); //weight zero must be prevented!!!
                    //global_i=i;
                    //cout << "i: " << trgdict->decode(trgdata->docword(s,i)) << "\n";
                    A[s][i][n][j]=LogGauss(D, W2V[srcdata->docword(s,j)],
                                           TM[trgdata->docword(s,i)].G[n].M,
                                           TM[trgdata->docword(s,i)].G[n].S)
                    +log(TM[trgdata->docword(s,i)].W[n])
                    +(i>0 || !use_null_word ?logf(1-NullProb):logf(NullProb))
                    +(i>0 || !use_null_word ?LogDistortion(delta):0);
                    
                    if (i==0 && n==0) //den must be initialized
                        den=A[s][i][n][j];
                    else
                        den=logsum(den,A[s][i][n][j]);
                }
            }
        //update local likelihood
        localLL[s]+=den;
        
        for (int i=0;i<trglen;i++)
            if ((use_null_word && i==0) || abs(i-j-1) <= distortion_window)
                for (int n=0;n<TM[trgdata->docword(s,i)].n;n++){
                    
                    assert(A[s][i][n][j]<= den);
                    
                    A[s][i][n][j]=expf(A[s][i][n][j]-den); // A is now a regular expected count
                    
                    if (A[s][i][n][j]<0.000000001) A[s][i][n][j]=0; //take mall risk of wrong normalization
                    
                    if (A[s][i][n][j]>0) TM[trgdata->docword(s,i)].G[n].eC++; //increase support set size
                    
                }
    }
    
    
    
}

void cswam::EstimateBeta(float &a, float &b,  float m,  float s){
    
    b = (s * m -s + m * m * m - 2 * m * m + m)/s;
    a = ( m * b )/(1-m);
}


void cswam::maximization(void *argv){
    
    long long d=(long long) argv;
    
    ShowProgress(d,D);
    
    if (d==D){
        //this thread is to maximize the global distortion model
        //Maximization step: Mean and variance of distortion model
        
        //Mean
        
        double totwdist=0, totdistprob=0, totnullprob=0, delta=0;
        for (int s=0;s<srcdata->numdoc();s++){
            for (int j=0;j<srcdata->doclen(s);j++)
                for (int i=0;i<trgdata->doclen(s);i++)
                    if ((use_null_word && i==0) || abs(i-j-1) <= distortion_window){
                        delta=Delta(i,j,trgdata->doclen(s),srcdata->doclen(s));
                        for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                            if (A[s][i][n][j]>0){
                                if (i>0 || !use_null_word){
                                    totwdist+=A[s][i][n][j]*delta;
                                    totdistprob+=A[s][i][n][j];
                                }
                                else{
                                    totnullprob+=A[s][i][n][j];
                                }
                            }
                    }
        }
        
        if (use_null_word && fix_null_prob==0)
            NullProb=(float)totnullprob/(totdistprob+totnullprob);
        
        if (distortion_mean && iter >0) //then update the mean
            DistMean=totwdist/totdistprob;
        
        
        //Variance
        if (distortion_var && iter >0){
            double  totwdeltadist=0;
            for (int s=0;s<srcdata->numdoc();s++)
                for (int i=1;i<trgdata->doclen(s);i++) //exclude i=0!
                    for (int j=0;j<srcdata->doclen(s);j++)
                        if (abs(i-j-1) <= distortion_window){
                            delta=Delta(i,j,trgdata->doclen(s),srcdata->doclen(s));
                            for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                                if (A[s][i][n][j]>0)
                                    totwdeltadist+=A[s][i][n][j] * (delta-DistMean) * (delta-DistMean);
                            
                        }
            
            DistVar=totwdeltadist/totdistprob;
        }
        
        cerr << "Dist: " << DistMean << " " << DistVar << "\n";
        
        if (use_null_word)
            cerr << "NullProb: " << NullProb << "\n";
        
        if (use_beta_distortion){
            cerr << "Beta A: " << DistA << " Beta B: " << DistB << "\n";
            EstimateBeta(DistA,DistB,DistMean,DistVar);
        }
        
    }
    else{
        //Maximization step: Mean;
        for (int s=0;s<srcdata->numdoc();s++)
            for (int j=0;j<srcdata->doclen(s);j++)
                for (int i=0;i<trgdata->doclen(s);i++)
                    if ((use_null_word && i==0) || abs(i-j-1) <= distortion_window)
                        for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                            if (A[s][i][n][j]>0)
                                TM[trgdata->docword(s,i)].G[n].M[d]+=A[s][i][n][j] * W2V[srcdata->docword(s,j)][d];
        
        //second pass
        for (int e=0;e<trgdict->size();e++)
            for (int n=0;n<TM[e].n;n++)
                if (Den[e][n]>0)
                    TM[e].G[n].M[d]/=Den[e][n]; //update the mean estimated
        
        if (train_variances){
            //Maximization step: Variance;
            
            for (int s=0;s<srcdata->numdoc();s++)
                for (int j=0;j<srcdata->doclen(s);j++)
                    for (int i=0;i<trgdata->doclen(s);i++)
                        if ((use_null_word && i==0) || abs(i-j-1) <= distortion_window)
                            for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                                if (A[s][i][n][j]>0)
                                    TM[trgdata->docword(s,i)].G[n].S[d]+=
                                    (A[s][i][n][j] *
                                     (W2V[srcdata->docword(s,j)][d]-TM[trgdata->docword(s,i)].G[n].M[d]) *
                                     (W2V[srcdata->docword(s,j)][d]-TM[trgdata->docword(s,i)].G[n].M[d])
                                     );
            
            //second pass
            for (int e=0;e<trgdict->size();e++)
                for (int n=0;n<TM[e].n;n++)
                    if (Den[e][n]>0){
                        TM[e].G[n].S[d]/=Den[e][n]; //might be too aggressive?
                        if (TM[e].G[n].S[d] < min_variance) TM[e].G[n].S[d]=min_variance; //improves generalization!
                    }
        }
    }
    
}


void cswam::expansion(void *argv){
    
    long long e=(long long) argv;
    for (int n=0;n<TM[e].n;n++){
        //get mean of variances
        float S=0; for (int d=0;d<D;d++) S+=TM[e].G[n].S[d]; S/=D;
        
        //variance treshold and population threshold
        float SThresh=5 * min_variance; float eCThresh=10;
        
        //show large support set and variances that do not reduce: more aggressive split
        if ((S/TM[e].G[n].mS) >= 0.95 &&      //mean variance does not reduce significantly
            TM[e].G[n].eC >= eCThresh  &&      //population is large
            S > SThresh) {                     //variance is large
            if (verbosity)
                cerr << "\n" << trgdict->decode(e) << " n= " << n << " (" << TM[e].n << ") Counts: "
                << TM[e].G[n].eC  << " mS: " << S << "\n";
            
            //expand: create new Gaussian after Gaussian n
            Gaussian *nG=new Gaussian[TM[e].n+1];
            float    *nW=new float[TM[e].n+1];
            memcpy((void *)nG,(const void *)TM[e].G, (n+1) * sizeof(Gaussian));
            memcpy((void *)nW,(const void *)TM[e].W, (n+1) * sizeof(float));
            if (n+1 < TM[e].n){
                memcpy((void *)&nG[n+2],(const void*)&TM[e].G[n+1],(TM[e].n-n-1) * sizeof(Gaussian));
                memcpy((void *)&nW[n+2],(const void*)&TM[e].W[n+1],(TM[e].n-n-1) * sizeof(float));
            }
            //initialize mean and variance vectors
            nG[n+1].M=new float[D];nG[n+1].S=new float[D];
            for (int d=0;d<D;d++){ //assign new means, keep old variances
               
                nG[n+1].M[d]=nG[n].M[d] + 2 * sqrt(nG[n].S[d]);
                nG[n].M[d]=nG[n].M[d] - 2 * sqrt(nG[n].S[d]);
              
                nG[n+1].S[d]=nG[n].S[d]=(2 * nG[n].S[d]); //enlarge a bit the variance: maybe better increase
            }
            nG[n+1].eC=nG[n].eC;
            nG[n+1].mS=nG[n].mS=S;
            
            //initialize weight vectors uniformly over n and n+1
            nW[n+1]=nW[n]/2;nW[n]=nW[n]/2;
            
            //update TM[e] structure
            TM[e].n++;
            delete [] TM[e].G;TM[e].G=nG;
            delete [] TM[e].W; TM[e].W=nW;
            
            //we increment loop variable by 1
            n++;
        }else{
            TM[e].G[n].mS=S;
        }
        
    }
    
}

float rl2(const float* a,const float*b, int d){
    float dist=0;
    float norm=0;
    for (int i=0;i<d;i++){
        dist=(a[i]-b[i])*(a[i]-b[i]);
        norm=a[i]*a[i];
    }
    return (norm>0?dist/norm:1);
}

float rl1(const float* a,const float*b, int d){
    float maxreldist=0; float reldist;
    for (int i=0;i<d;i++){
        reldist=(abs(a[i]-b[i])/a[i]);
        if (reldist>maxreldist) maxreldist=reldist;
    }
    return maxreldist;
}

float al1(const float* a,const float*b, int d){
    float maxdist=0; float dist;
    for (int i=0;i<d;i++){
        dist=abs(a[i]-b[i]);
        if (dist>maxdist) maxdist=dist;
    }
    return maxdist;
}

void cswam::contraction(void *argv){
    
    long long e=(long long) argv;
    
    float min_std=sqrt(min_variance);
    float min_weight=0.01;
    
    for (int n=0;n<TM[e].n;n++){
        int n1=0;
        // look if the component overlaps with some of the previous ones
        float max_dist=1;
        for (n1=0;n1<n;n1++) if ((max_dist=al1(TM[e].G[n].M,TM[e].G[n1].M,D))< min_std) break;

        //remove insignificant and overlapping gaussians (relative distance below minimum variance
        if (TM[e].W[n] < min_weight || max_dist < min_std) { //eliminate this component
            assert(TM[e].n>1);
            if (verbosity) cerr << "\n" << trgdict->decode(e) << " n= " << n << " Weight: " << TM[e].W[n] <<  " Dist= " << max_dist << "\n";
            //expand: create new mixture model with n-1 components
            Gaussian *nG=new Gaussian[TM[e].n-1];
            float    *nW=new float[TM[e].n-1];
            if (n>0){ //copy all entries before n
                memcpy((void *)nG,(const void *)TM[e].G, n * sizeof(Gaussian));
                memcpy((void *)nW,(const void *)TM[e].W, n * sizeof(float));
            }
            if (n+1 < TM[e].n){  //copy all entries after
                memcpy((void *)&nG[n],(const void*)&TM[e].G[n+1],(TM[e].n-n-1) * sizeof(Gaussian));
                memcpy((void *)&nW[n],(const void*)&TM[e].W[n+1],(TM[e].n-n-1) * sizeof(float));
            }
            
            //don't need to normalized weights!
            if (max_dist < min_std)// this is the gaussian overlapping case
                nW[n1]+=TM[e].W[n];  //the left gaussian inherits the weight
            
            //update TM[e] structure
            TM[e].n--;n--;
            delete [] TM[e].G;TM[e].G=nG;
            delete [] TM[e].W; TM[e].W=nW;
        }
    }
    
    //re-normalize weights
    float totw=0;
    for (int n=0;n<TM[e].n;n++){totw+=TM[e].W[n]; assert(TM[e].W[n] > 0.0001);}
    for (int n=0;n<TM[e].n;n++){TM[e].W[n]/=totw;};
}

int cswam::train(char *srctrainfile, char*trgtrainfile,char *modelfile, int maxiter,int threads){
    
    //initialize model    
    initModel(modelfile); //this might change the dictionary!
    
    //Load training data

    srcdata=new doc(srcdict,srctrainfile);
    trgdata=new doc(trgdict,trgtrainfile,use_null_word); //use null word

   
    iter=0;
    
    cerr << "Starting training";
    threadpool thpool=thpool_init(threads);
    int numtasks=trgdict->size()>trgdata->numdoc()?trgdict->size():trgdata->numdoc();
    task *t=new task[numtasks];
    assert(numtasks>D); //multi-threading also distributed over D
    
    
    //support variable to compute likelihood
    localLL=new float[srcdata->numdoc()];
    
    while (iter < maxiter){
        
        cerr << "\nIteration: " << ++iter <<  "\n";
        
        initAlphaDen();
        
        //reset support set size
        for (int e=0;e<trgdict->size();e++)
            for (int n=0;n<TM[e].n;n++)  TM[e].G[n].eC=0; //will be updated in E-step
        
        
        cerr << "E-step: ";
        //compute expected counts in each single sentence
        for (long long  s=0;s<srcdata->numdoc();s++){
            //prepare and assign tasks to threads
            t[s].ctx=this; t[s].argv=(void *)s;
            thpool_add_work(thpool, &cswam::expected_counts_helper, (void *)&t[s]);
            
        }
        //join all threads
        thpool_wait(thpool);
        
        
        //Reset model before update
        for (int e=0;e <trgdict->size();e++)
            for (int n=0;n<TM[e].n;n++){
                memset(TM[e].G[n].M,0,D * sizeof (float));
                if (train_variances)
                    memset(TM[e].G[n].S,0,D * sizeof (float));
            }
        
        for (int e=0;e<trgdict->size();e++)
            memset(Den[e],0,TM[e].n * sizeof(float));
        
        cswam_LL=0; //compute LL of current model
        //compute normalization term for each target word
        for (int s=0;s<srcdata->numdoc();s++){
            cswam_LL+=localLL[s];
            for (int i=0;i<trgdata->doclen(s);i++)
                for (int n=0;n<TM[trgdata->docword(s,i)].n;n++)
                    for (int j=0;j<srcdata->doclen(s);j++)
                        Den[trgdata->docword(s,i)][n]+=A[s][i][n][j];
        }
        
        cerr << "LL = " << cswam_LL << "\n";
        
        
        cerr << "M-step: ";
        for (long long d=0;d<=D;d++){  //include special job d=D for distortion model
            t[d].ctx=this; t[d].argv=(void *)d;
            thpool_add_work(thpool, &cswam::maximization_helper, (void *)&t[d]);
        }
                
        //join all threads
        thpool_wait(thpool);
        
        //some checks of the models: fix degenerate models
        for (int e=0;e<trgdict->size();e++)
            if (e!=trgEoD)
                for (int n=0;n<TM[e].n;n++)
                                       if (!Den[e][n]){
                        if (verbosity)
                            cerr << "\nRisk of degenerate model. Word: " << trgdict->decode(e) << " n: " << n << " eC:" << TM[e].G[n].eC << "\n";
                        for (int d=0;d<D;d++) TM[e].G[n].S[d]=SSEED * min_variance;
                    }
                
       
//            if (trgdict->encode("bege")==e){
//                cerr << "bege " << " mS: " << TM[e].G[0].mS << " n: " << TM[e].n << " eC " << TM[e].G[0].eC << "\n";
//                cerr << "M:"; for (int d=0;d<10;d++) cerr << " " << TM[e].G[0].M[d]; cerr << "\n";
//                cerr << "S:"; for (int d=0;d<10;d++) cerr << " " << TM[e].G[0].S[d]; cerr << "\n";
//            }
//        }
        
        //update the weight estimates: ne need of multithreading
        float totW; int ngauss=0;
        for (int e=0;e<trgdict->size();e++){
            totW=0;
            for (int n=0;n<TM[e].n;n++){ totW+=Den[e][n]; ngauss++;}
            if (totW>0)
                for (int n=0;n<TM[e].n;n++) TM[e].W[n]=Den[e][n]/totW;
        }
        cerr << "Num Gaussians: " << ngauss << "\n";
        
        if (iter > 1 || incremental_train ){
            
            freeAlphaDen(); //needs to be reallocated as models might change
            
            cerr << "\nP-step: ";
            for (long long e=0;e<trgdict->size();e++){
                //check if to decrease number of gaussians per target word
                t[e].ctx=this; t[e].argv=(void *)e;
                thpool_add_work(thpool, &cswam::contraction_helper, (void *)&t[e]);
            }
            //join all threads
            thpool_wait(thpool);
            
            cerr << "\nS-step: ";
            for (long long e=0;e<trgdict->size();e++){
                //check if to increase number of gaussians per target word
                t[e].ctx=this; t[e].argv=(void *)e;
                thpool_add_work(thpool, &cswam::expansion_helper, (void *)&t[e]);
            }
            //join all threads
            thpool_wait(thpool);
            
            
        }
        
       
        if (srcdata->numdoc()>10000) system("date");
        
        saveModel(modelfile);

    }
    
   // for (int e=0;e<trgdict->size();e++)
   //     for (int d=0;d<D;d++)
   //         cout << trgdict->decode(e) << " S: " << S[e][d] << " M: " << M[e][d]<< "\n";

    //destroy thread pool
    thpool_destroy(thpool);
  
    freeAlphaDen();

    delete srcdata; delete trgdata;
    delete [] t; delete [] localLL;
    
    return 1;
}



void cswam::aligner(void *argv){
    long long s=(long long) argv;
    static float maxfloat=std::numeric_limits<float>::max();
    
    
    if (! (s % 10000)) {cerr << ".";cerr.flush();}
    //fprintf(stderr,"Thread: %lu  Document: %d  (out of %d)\n",(long)pthread_self(),s,srcdata->numdoc());
    
    int trglen=trgdata->doclen(s); // length of target sentence
    int srclen=srcdata->doclen(s); //length of source sentence
    
    assert(trglen<MAX_LINE);
    
    //Viterbi alignment: find the most probable alignment for source
    float score; float best_score;int best_i;float sum=0;
    
    bool some_not_null=false; int first_target=0;
    
    for (int j=0;j<srclen;j++){
        //cout << "src: " << srcdict->decode(srcdata->docword(s,j)) << "\n";
        
        best_score=-maxfloat;best_i=0;
        
        for (int i=first_target;i<trglen;i++)
            if ((use_null_word && i==0) || abs(i-j-1) <= distortion_window){
                //cout << "tgt: " << trgdict->decode(trgdata->docword(s,i)) << " ";
                
                for (int n=0;n<TM[trgdata->docword(s,i)].n;n++){
                    score=LogGauss(D,
                                   W2V[srcdata->docword(s,j)],
                                   TM[trgdata->docword(s,i)].G[n].M,
                                   TM[trgdata->docword(s,i)].G[n].S)+log(TM[trgdata->docword(s,i)].W[n]);
                    if (n==0) sum=score;
                    else sum=logsum(sum,score);
                } //completed mixture score
                
                if (distortion_var || distortion_mean){
                    if (i>0 ||!use_null_word){
                        float d=Delta(i,j,trglen,srclen);
                        sum+=logf(1-NullProb) + LogDistortion(d);
                    }
                    else
                        if (use_null_word ) sum+=logf(NullProb);
                }
                else //use plain distortion model
                    if (i>0){
                        if (i - (use_null_word?1:0) > j )
                            sum-=(i- (use_null_word?1:0) -j);
                        else if  (i - (use_null_word?1:0) < j )
                            sum-=(j - i + (use_null_word?1:0));
                    }
                //add distortion score now
                
                //cout << "score: " << sum << "\n";
                //  cout << "\t " << srcdict->decode(srcdata->docword(s,j)) << "  " << dist << "\n";
                //if (dist > -50) score=(float)exp(-dist)/norm;
                if (sum > best_score){
                    best_score=sum;
                    best_i=i;
                    if ((!use_null_word || best_i>0) && !some_not_null) some_not_null=true;
                }
            }
        //cout << "best_i: " << best_i << "\n";
        
        alignments[s % bucket][j]=best_i;
        
        if (j==(srclen-1) && !some_not_null){
            j=-1; //restart loop and remove null word from options
            first_target=1;
            some_not_null=true; //make sure to pass this check next time
        }
    }
    
}


int cswam::test(char *srctestfile, char *trgtestfile, char* modelfile, char* alignfile,int threads){
    
    {mfstream out(alignfile,ios::out);} //empty the file
    
    initModel(modelfile);
    
    if (!distortion_mean){
        if (use_beta_distortion){
            cerr << "ERROR: cannot test with beta distribution without mean\n";
            return 0;
        }
        DistMean=0; //force mean to zero
    }
    
    //Load training data
    srcdata=new doc(srcdict,srctestfile);
    trgdata=new doc(trgdict,trgtestfile,use_null_word);
    assert(srcdata->numdoc()==trgdata->numdoc());
    
   
    bucket=BUCKET; //initialize the bucket size
    
    alignments=new int* [BUCKET];
    for (int s=0;s<BUCKET;s++)
        alignments[s]=new int[MAX_LINE];
    
    threadpool thpool=thpool_init(threads);
    task *t=new task[BUCKET];
    
    cerr << "Start alignment\n";
    
    for (long long s=0;s<srcdata->numdoc();s++){
        
        t[s % BUCKET].ctx=this; t[s % BUCKET].argv=(void *)s;
        thpool_add_work(thpool, &cswam::aligner_helper, (void *)&t[s % BUCKET]);
        

        if (((s % BUCKET) == (BUCKET-1)) || (s==(srcdata->numdoc()-1)) ){
            //join all threads
            thpool_wait(thpool);
            
            //cerr << "Start printing\n";
            
            if ((s % BUCKET) != (BUCKET-1))
                    bucket=srcdata->numdoc() % bucket; //last bucket at end of file
            
                mfstream out(alignfile,ios::out | ios::app);
                
                for (int b=0;b<bucket;b++){ //includes the eof case of
                    //out << "Sentence: " << s-bucket+1+b;
                    bool first=true;
                    for (int j=0; j<srcdata->doclen(s-bucket+1+b); j++)
                        if (!use_null_word || alignments[b][j]>0){
                            //print target using 0 for first actual word
                            out << (first?"":" ") << j << "-" << alignments[b][j]-(use_null_word?1:0);
                            first=false;
                        }
                    out << "\n";
                }
        }
        
    }
    
    
    //destroy thread pool
    thpool_destroy(thpool);
    
    delete [] t;
    for (int s=0;s<BUCKET;s++) delete [] alignments[s];delete [] alignments;
    delete srcdata; delete trgdata;
    return 1;
}

//find for each target word a list of associated source words

typedef std::pair <int,float> mientry;  //pair type containing src word and mi score
bool myrank (Friend a,Friend b) { return (a.score > b.score ); }


//void cswam::findfriends(FriendList* friends){
//    
//    typedef std::unordered_map<int, int> src_map;
//    src_map* table= new src_map[trgdict->size()];
//    
//    //  amap["def"][7] = 2.2;
//    //  std::cout << amap["abc"][12] << '\n';
//    //  std::cout << amap["def"][7] << '\n';
//    
//    
//    int *srcfreq=new int[srcdict->size()];
//    int *trgfreq=new int[trgdict->size()];
//    int totfreq=0;
//    int minfreq=10;
//    
//    cerr << "collecting co-occurrences\n";
//    for (int s=0;s<srcdata->numdoc();s++){
//        
//        int trglen=trgdata->doclen(s); // length of target sentence
//        int srclen=srcdata->doclen(s); //length of source sentence
//        
//        int frac=(s * 1000)/srcdata->numdoc();
//        if (!(frac % 10)) fprintf(stderr,"%02d\b\b",frac/10);
//        
//        for (int i=0;i<trglen;i++){
//            int trg=trgdata->docword(s,i);
//            float trgdictfreq=trgdict->freq(trg);
//            if (trgdict->freq(trg)>=10){
//                for (int j=0;j<srclen;j++){
//                    int src=srcdata->docword(s,j);
//                    float freqratio=srcdict->freq(src)/trgdictfreq;
//                    if (srcdict->freq(src)>=minfreq && freqratio <= 10 && freqratio >= 0.1){
//                        table[trg][src]++;
//                        totfreq++;
//                        srcfreq[src]++;
//                        trgfreq[trg]++;
//                    }
//                }
//            }
//        }
//    }
//    
//    cerr << "computing mutual information\n";
//    Friend mie; FriendList mivec;
//    
//    
//    for (int i = 0; i < trgdict->size(); i++){
//        
//        int frac=(i * 1000)/trgdict->size();
//        if (!(frac % 10)) fprintf(stderr,"%02d\b\b",frac/10);
//        
//        mivec.clear();
//        for (auto jtr = table[i].begin(); jtr !=  table[i].end();jtr++){
//            int j=(*jtr).first; int freq=(*jtr).second;
//            float freqratio=(float)srcdict->freq(j)/(float)trgdict->freq(i);
//            if (freq>minfreq){  // && freqratio < 10 && freqratio > 0.1){
//                //compute mutual information
//                float mutualinfo=
//                logf(freq/(float)trgfreq[i]) - log((float)srcfreq[j]/totfreq);
//                mutualinfo/=log(2);
//                mie.word=j; mie.score=mutualinfo;
//                mivec.push_back(mie);
//            }
//        }
//        if (mivec.size()>0){
//            std::sort(mivec.begin(),mivec.end(),myrank);
//            //sort the vector and take the top log(10)
//            int count=0;
//            for (auto jtr = mivec.begin(); jtr !=  mivec.end();jtr++){
//                //int j=(*jtr).word; float mutualinfo=(*jtr).score;
//                friends[i].push_back(*jtr);
//                //cout << trgdict->decode(i) << " " << srcdict->decode(j) << " " << mutualinfo << endl;
//                //if (++count >=50) break;
//            }
//            
//        }
//    }
//    
//    
//}



void cswam::M1_ecounts(void *argv){
    long long s=(long long) argv;

    int b=s % threads; //value of the actual bucket
    int trglen=trgdata->doclen(s); // length of target sentence
    int srclen=srcdata->doclen(s); //length of source sentence
    float pef=0;
    
    ShowProgress(s,srcdata->numdoc());
    
    float lowprob=0.0000001;
    
    for (int j=0;j<srclen;j++){
        int f=srcdata->docword(s,j);
        if (srcdict->freq(f)>=minfreq){
            float t=0;
            for (int i=0;i<trglen;i++){
                int e=trgdata->docword(s,i);
                if (trgdict->freq(e)>=minfreq && (i==0 || abs(i-j-1) <= distortion_window) && prob[e][f]>lowprob)
                    t+=prob[e][f];
            }
            for (int i=0;i<trglen;i++){
                int e=trgdata->docword(s,i);
                if (trgdict->freq(e)>=minfreq && (i==0 || abs(i-j-1) <= distortion_window) && prob[e][f]>lowprob){
                    pef=prob[e][f]/t;
                    loc_efcounts[b][e][f]+=pef;
                    loc_ecounts[b][e]+=pef;
                }
            }
        }
    }
    
}

void cswam::M1_update(void *argv){
    long long e=(long long) argv;
    
    ShowProgress(e,trgdict->size());
    
//    for (auto jtr = efcounts[e].begin(); jtr != efcounts[e].end();jtr++){
    for (src_map::iterator jtr = efcounts[e].begin(); jtr !=  efcounts[e].end();jtr++){
        int f=(*jtr).first;
        prob[e][f]=efcounts[e][f]/ecounts[e];
    }
}

void cswam::M1_collect(void *argv){
    long long e=(long long) argv;

    ShowProgress(e,trgdict->size());
    
    for (int b=0;b<threads;b++){
        ecounts[e]+=loc_ecounts[b][e];
        loc_ecounts[b][e]=0; //reset local count
//        for (auto jtr = loc_efcounts[b][e].begin(); jtr != loc_efcounts[b][e].end();jtr++){
        for (src_map::iterator jtr = loc_efcounts[b][e].begin(); jtr !=  loc_efcounts[b][e].end();jtr++){
            int f=(*jtr).first;
            efcounts[e][f]+=loc_efcounts[b][e][f];
        }
        loc_efcounts[b][e].clear(); //reset local counts
    }
}


void cswam::M1_clearcounts(bool clearmem){

    if (efcounts==NULL){
        cerr << "allocating thread local structures\n";
        //allocate thread safe structures
        loc_efcounts=new src_map*[threads];
        loc_ecounts=new float*[threads];
        for (int b=0;b<threads;b++){
            loc_efcounts[b]=new src_map[trgdict->size()];
            loc_ecounts[b]=new float[trgdict->size()];
        }
        cerr << "allocating global count structures\n";
        //allocate the global count structures
        efcounts=new src_map[trgdict->size()];
        ecounts=new float[trgdict->size()];
    }
    
    
    if (clearmem){
        for (int b=0;b<threads;b++){
            delete [] loc_efcounts[b];
            delete [] loc_ecounts[b];
        }
        delete [] loc_efcounts; delete [] loc_ecounts;
        delete [] efcounts; delete [] ecounts;
    }else{
       // cerr << "resetting expected counts\n";
        for (int e = 0; e < trgdict->size(); e++){
            efcounts[e].clear();
            memset(ecounts,0,sizeof(int)*trgdict->size());
        }
        //local expected counts are reset in main loop
    }

}


void cswam::findfriends(FriendList* friends){
           
    //allocate the global prob table
    prob= new src_map[trgdict->size()];
    
    //allocate thread safe structures
    M1_clearcounts(false);
    
    //prepare thread pool
    threadpool thpool=thpool_init(threads);
    task *t=new task[trgdict->size()>threads?trgdict->size():threads];
    
    float minprob=0.01;
    
    cerr << "initializing M1\n";
    for (int s=0;s<srcdata->numdoc();s++){
        int trglen=trgdata->doclen(s); // length of target sentence
        int srclen=srcdata->doclen(s); //length of source sentence
        
        int frac=(s * 1000)/srcdata->numdoc();
        if (!(frac % 10)) fprintf(stderr,"%02d\b\b",frac/10);
        
        for (int j=0;j<srclen;j++){
            int f=srcdata->docword(s,j);
            if (srcdict->freq(f)>=minfreq){
                for (int i=0;i<trglen;i++){
                    int e=trgdata->docword(s,i);
                    if (trgdict->freq(e)>=minfreq && (i==0 || abs(i-j-1) <= distortion_window))
                        prob[e][f]=1;
                }
            }
        }
    }
    
    cerr << "training M1\n";
    for (int it=0;it<M1iter;it++){
        
        cerr << "it: " << it+1;
        M1_clearcounts(false);
        
        //compute expected counts
        for (long long s=0;s<srcdata->numdoc();s++){
            
            t[s % threads].ctx=this; t[s % threads].argv=(void *)s;
            thpool_add_work(thpool, &cswam::M1_ecounts_helper,(void *)&t[s % threads]);
            
            if (((s % threads) == (threads-1)) || (s==(srcdata->numdoc()-1)))
                thpool_wait(thpool);//join all threads
        }
        
        //update the global counts
        for (long long e = 0; e < trgdict->size(); e++){
                t[e].ctx=this; t[e].argv=(void *)e;
                thpool_add_work(thpool, &cswam::M1_collect_helper,(void *)&t[e]);
        }
        thpool_wait(thpool);//join all threads
        
        //update probabilities
        for (long long e = 0; e < trgdict->size(); e++){
            t[e].ctx=this; t[e].argv=(void *)e;
            thpool_add_work(thpool, &cswam::M1_update_helper,(void *)&t[e]);
        }
        
        thpool_wait(thpool); //join all threads
    }
    
    cerr << "computing candidates\n";
    Friend f;FriendList fv;
    
    for (int e = 0; e < trgdict->size(); e++){
        
        ShowProgress(e,trgdict->size());
     
        fv.clear();
        //save in a vector and compute entropy
        float H=0;
//        for (auto jtr = prob[e].begin(); jtr !=  prob[e].end();jtr++){
        for (src_map::iterator jtr = prob[e].begin(); jtr !=  prob[e].end();jtr++){
            f.word=(*jtr).first; f.score=(*jtr).second;
            assert(f.score>=0 && f.score<=1);
            if (f.score>0)
                H-=f.score * logf(f.score);
            if (f.score >= minprob)  //never include options with prob < minprob
                fv.push_back(f);
        }
        
        std::sort(fv.begin(),fv.end(),myrank);
        int PP=round(expf(H)); //compute perplexity

        cout << trgdict->decode(e) << " # friends: " << fv.size() << " PP " << PP << endl;
               int count=0;
//        for (auto jtr = fv.begin(); jtr !=  fv.end();jtr++){
        for (FriendList::iterator jtr = fv.begin(); jtr !=  fv.end();jtr++){
            friends[e].push_back(*jtr);
            //if (verbosity)
            cout << trgdict->decode(e) << " " << srcdict->decode((*jtr).word) << " " << (*jtr).score << endl;
            if (++count >= PP) break;
        }
    }
    
    //destroy thread pool
    thpool_destroy(thpool); delete [] t;
    
    M1_clearcounts(true);
    
    delete [] prob;
}

} //namespace irstlm

