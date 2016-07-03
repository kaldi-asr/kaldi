#include "head.h"
#include "helper.h"
#include "fileops.h"
#include "DataType.h"
#include <omp.h>

namespace cuedrnnlm {

class matrix
{
private:
    Real* host_data;
    size_t nrows;
    size_t ncols;
    size_t size;
public:
    matrix ():host_data(NULL), nrows(0), ncols(0)
    {}
    matrix (size_t nr, size_t nc)
    {
        nrows = nr;
        ncols = nc;
        size = sizeof(Real) * ncols * nrows;
        host_data = (Real *) malloc (size);
    }
    ~matrix ()
    {
        if (host_data)
        {
            free (host_data);
            host_data = NULL;
        }
    }
    size_t Sizeof ()
    {
        return (nrows * ncols * sizeof(Real));
    }
    size_t nelem ()
    {
        return (nrows * ncols);
    }
    // asign value on CPU
    void assignhostvalue (size_t i, size_t j, Real v)
    {
        host_data[i+j*nrows] = v;
    }
    void addhostvalue (size_t i, size_t j, Real v)
    {
        host_data[i+j*nrows] += v;
    }
    Real fetchhostvalue (size_t i, size_t j)
    {
        return host_data[i+j*nrows];
    }

    void setnrows (size_t nr)
    {
        nrows = nr;
    }
    void setncols (size_t nc)
    {
        ncols = nc;
    }
    size_t rows ()
    {
        return nrows;
    }
    size_t cols ()
    {
        return ncols;
    }
    void freemem ()
    {
        free (host_data);
        ncols = 0;
        nrows = 0;
        size = 0;
    }
    Real& operator() (int i, int j) const
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        return host_data[i + j*nrows];
    }
    const Real& operator() (int i, int j)
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        return host_data[i + j*nrows];
    }
    Real* gethostdataptr ()
    {
        return host_data;
    }
    Real *gethostdataptr(int i, int j)
    {
        return &host_data[i+j*nrows];
    }

    // initialize all element (both GPU and CPU) in matrx with v
    void initmatrix (int v = 0)
    {
        memset (host_data, v, Sizeof());
    }
    void hostrelu (float ratio)
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            if (host_data[i] > 0)
            {
                host_data[i] *= ratio;
            }
            else
            {
                host_data[i] = 0;
            }
        }
    }
    void hostsigmoid()
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            host_data[i] = 1/(1 + exp(-host_data[i]));
        }
    }

    void hostsoftmax()
    {
        int a, maxi;
        float v, norm, maxv = 1e-8;
        assert (ncols == 1);
        maxv = 1e-10;
        for (a=0; a<nrows; a++)
        {
            v = host_data[a];
            if (v > maxv)
            {
                maxv = v;
                maxi = a;
            }
        }
        norm = 0;
        for (a=0; a<nrows; a++)
        {
            v = host_data[a] - maxv;
            host_data[a] = exp(v);
            norm += host_data[a];
        }
        for (a=0; a<nrows; a++)
        {
            v = host_data[a] / norm;
            host_data[a] = v;
        }
    }

    void hostpartsoftmax(int swordid, int ewordid)
    {
        int a, maxi;
        float v, norm, maxv = 1e-8;
        assert (ncols == 1);
        maxv = 1e-10;
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a];
            if (v > maxv)
            {
                maxv = v;
                maxi = a;
            }
        }
        norm = 0;
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a] - maxv;
            host_data[a] = exp(v);
            norm += host_data[a];
        }
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a] / norm;
            host_data[a] = v;
        }
    }

    void random(float min, float max)
    {
        int i, j;
        float v;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                v = randomv(min, max) + randomv(min,max) + randomv(min, max);
                host_data[i+j*nrows] = v;
            }
        }
    }
};

class RNNLM
{
protected:
    string inmodelfile, trainfile, validfile, nglmstfile,
           testfile, inputwlist, outputwlist, feafile;
    vector<int> &layersizes;
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
    RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1, vector<int> &lsizes, int fvocsize, bool bformat, int debuglevel);

    ~RNNLM();

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

    void setLognormConst (float v)   {lognormconst = v;}
    void setNthread (int n)         {nthread = n;}
    void setLmscale (float v)       {lmscale = v;}
    void setIp (float v)            {ip = v;}
    void setFullVocsize (int n);
    void copyRecurrentAc ();
    void ResetRechist();
    float forword (int prevword, int curword);
    void ReadUnigramFile (string unigramfile);
    void matrixXvector (float *ac0, float *wgt1, float *ac1, int nrow, int ncol);
    void allocMem (vector<int> &layersizes);

    // functions for using additional feature file in input layer
    void ReadFeaFile(string filestr);
    void setFeafile (string str)  {feafile = str;}

    // for Kaldi integration
    RNNLM(vector<int> &lsizes, int fvocsize);
    void loadfresh (const string &inmodelfile_1, const string &inputwlist_1, const string &outputwlist_1, bool bformat, int mbsize, int debuglevel);

    int getOneHiddenLayerSize();
    bool calLogProb(string input, int fullvocsize, float &logp, float &ppl);
    float computeConditionalLogprob(std::string current_word, const std::vector<std::string> &history_words, const std::vector<float> &context_in, std::vector<float> *context_out);
    void restoreContextFromVector(const std::vector<float> &context_in);
    void saveContextToVector(std::vector<float> *context_out);
    void setFullVocabSize(int fvocsize);

};

} // end namespace

