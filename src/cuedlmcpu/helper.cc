#ifndef __HEAD_HELPER__
#define __HEAD_HELPER__
#include "head.h"

void printusage(char *str)
{
    printf ("Usage of command \"%s\"\n", str);
    printf ("Function:\n");
    printf ("%35s\t%s\n", "-ppl                     :",                      "RNNLM evaluation for perplexity (CPU supported)");
    printf ("%35s\t%s\n", "-nbest                   :",                    "RNNLM evaluation for N best rescoring (CPU supported)" );
    printf ("Configuration:\n");
    printf ("%35s\t%s\n", "-validfile   <string>    :",     "specify the valid file for RNNLM training");
    printf ("%35s\t%s\n", "-testfile    <string>    :",     "specify the test file for RNNLM evaluation");
    printf ("%35s\t%s\n", "-feafile     <string>    :",     "specify the feature matrix file");
    printf ("%35s\t%s\n", "-inputwlist  <string>    :",     "specify the input word list for RNNLM training");
    printf ("%35s\t%s\n", "-outputwlist <string>    :",     "specify the output word list for RNNLM training");
    printf ("%35s\t%s\n", "-lognormconst <float>       :",        "specify the log norm const for NCE training and evaluation without normalization (default: -1.0)");
    printf ("%35s\t%s\n", "-lambda      <float>     :",      "specify the interpolation weight for RNNLM when interpolating with N-Gram LM (default: 0.5)");
    printf ("%35s\t%s\n", "-debug       <int>       :",        "specify the debug level (default: 1)");
    printf ("%35s\t%s\n", "-nthread     <int>       :",        "specify the number of thread for computation (default: 1)");
    printf ("%35s\t%s\n", "-readmodel   <string>    :",     "specify the RNNLM model to be read");
    printf ("%35s\t%s\n", "-fullvocsize <int>       :",        "specify the full vocabulary size, all OOS words will share the probability");
    printf ("%35s\t%s\n", "-lmscale <float>         :",        "specify the lmscale, used for nbest rescoring");
    printf ("%35s\t%s\n", "-ip <float>              :",        "specify the insertion penalty, used for nbest rescoring");
    printf ("%35s\t%s\n", "-binformat               :",                "specify the model will be read or write with binary format (default: false)");
    printf ("%35s\t%s\n", "-nglmstfile  <string>    :",     "specify the ngram lm stream file for interpolation");
    printf ("\nexample:\n");
    printf ("%s -ppl -readmodel h200.mb64/rnnlm.txt -testfile data/test.dat -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist -nglmstfile ng.st -lambda 0.5 -debug 2\n", str);
    printf ("%s -nbest -readmodel h200.mb64/rnnlm.txt.nbest -testfile data/test.dat  -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist -nglmstfile ng.st -lambda 0.5 -debug 2\n", str);
}


bool isEmpty(string str)
{
    if (str == "EMPTY")     return true;
    else                    return false;
}

int string2int (string str)
{
    return atoi (str.c_str());
}

float string2float (string str)
{
    return atof (str.c_str());
}

void parseArray (string str, vector<int> &layersizes)
{
    int pos;
   layersizes.clear();
   while (str.size() > 0)
   {
       pos = str.find_first_of(':');
       if (pos ==  string::npos)        break;
       string substr = str.substr(0, pos);
       layersizes.push_back(atoi(substr.c_str()));
       str = str.substr (pos+1);
   }
   layersizes.push_back(atoi(str.c_str()));
}

float randomv(float min, float max)
{
    return rand()/(Real)RAND_MAX*(max-min)+min;
}

float gaussrandv(float mean, float var)
{
    float v1, v2, s;
    int phase  = 0;
    double x;
    if (0 == phase)
    {
        do
        {
            float u1 = (float)rand()/RAND_MAX;
            float u2 = (float)rand()/RAND_MAX;

            v1 = 2 * u1 - 1;
            v2 = 2 * u2 - 1;
            s = v1 * v1 + v2 * v2;
        } while ( 1 <= s || 0 == s);
        x = v1 * sqrt(-2 * log(s) / s);
    }
    else
    {
        x = v2 * sqrt(-2 * log(s) / s);
    }
    phase = 1 - phase;
    x = var*x+mean;
    return x;
}

int getline (char *line, int &max_words_line, FILE *&fptr)
{
    int i=0;
    char ch;
    while (!feof(fptr))
    {
        ch = fgetc(fptr);
        if (ch == ' ' && i==0)
        {
            continue;
        }
        line[i++] = ch;
        if (ch == '\n')
        {
            break;
        }
    }
    line[i] = 0;
    return i;
}

// log(exp(x) + exp(y))
float logadd (float x, float y)
{
    if (x > y)
    {
        return (x + log(1+exp(y-x)));
    }
    else
    {
        return (y + log(1+exp(x-y)));
    }
}




#endif
