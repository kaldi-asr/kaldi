#include "cued-rnnlm-lib.h"

namespace cued_rnnlm {

void printusage(char *str) {
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

bool isEmpty(string str) {
  if (str == "EMPTY")     return true; // TODO(hxu) this is very weird
  else                    return false;
}

int string2int (string str) {
  return atoi (str.c_str());
}

float string2float (string str) {
  return atof (str.c_str());
}

void parseArray (string str, vector<int> &layersizes) {
  int pos;
  layersizes.clear();
  while (str.size() > 0) {
    pos = str.find_first_of(':');
    if (pos ==  string::npos)
      break;
    string substr = str.substr(0, pos);
    layersizes.push_back(atoi(substr.c_str()));
    str = str.substr (pos+1);
  }
  layersizes.push_back(atoi(str.c_str()));
}

float randomv(float min, float max) {
  return rand() / (real)RAND_MAX * (max-min) + min;
}

float gaussrandv(float mean, float var) {
  float v1, v2 = 1.0, s = 1.0;
  int phase  = 0;
  double x;
  if (0 == phase) {
    do {
      float u1 = (float)rand() / RAND_MAX;
      float u2 = (float)rand() / RAND_MAX;

      v1 = 2 * u1 - 1;
      v2 = 2 * u2 - 1;
      s = v1 * v1 + v2 * v2;
    } while ( 1 <= s || 0 == s);
    x = v1 * sqrt(-2 * log(s) / s);
  }
  else {
    x = v2 * sqrt(-2 * log(s) / s);
  }
  phase = 1 - phase;
  x = var * x + mean;
  return x;
}

int getline (char *line, int &max_words_line, FILE *&fptr) {
  int i=0;
  char ch;
  while (!feof(fptr)) {
    ch = fgetc(fptr);
    if (ch == ' ' && i==0) {
      continue;
    }
    line[i++] = ch;
    if (ch == '\n') {
      break;
    }
  }
  line[i] = 0;
  return i;
}

// log(exp(x) + exp(y))
float logadd (float x, float y) {
  if (x > y) {
    return (x + log(1 + exp(y - x)));
  }
  else {
    return (y + log(1 + exp(x - y)));
  }
}

float random(float min, float max) {
    return rand() / (real)RAND_MAX * (max - min) + min;
}

void RNNLM::init() {
  lognormconst    = 0;
  lambda          = 0.5;
  version         = 0.1;
  iter            = 0;
  num_layer       = 0;
  wordcn          = 0;
  validwordcnt    = 0;
  counter         = 0;
  num_oosword     = 0;
  num_fea         = 0;
  dim_fea         = 0;
  nodetype        = 0;                // sigmoid is applied by default
  reluratio       = 0.5;
  resetAc         = NULL;
  lognorms        = NULL;
  layer0_fea      = NULL;
  neu0_ac_fea     = NULL;
  feamatrix       = NULL;
  word2class      = NULL;
  classinfo       = NULL;
  layerN_class    = NULL;
  neuN_ac_class   = NULL;
  nclass          = 0;
  nthread		    = 0;
  lmscale         = 12.0;
  ip              = 0;        // insertion penalty
  minibatch       = 1;
}

RNNLM::~RNNLM() {
  int i;
  if (neu0_ac_hist)      {delete neu0_ac_hist; neu0_ac_hist=NULL;}
  if (layer0_hist)       {delete layer0_hist; layer0_hist=NULL;}
  if (lognorms)          {delete lognorms; lognorms = NULL;}
  for (i = 0; i < num_layer; i++) {
    delete layers[i];
    delete neu_ac[i];
    layers[i] = NULL;
    neu_ac[i] = NULL;
  }
  delete neu_ac[num_layer];
  neu_ac[num_layer] = NULL;
  if (resetAc)    {delete [] resetAc; resetAc = NULL; }
  if (layer0_fea) {delete layer0_fea; layer0_fea=NULL;}
  if (neu0_ac_fea) {delete neu0_ac_fea; neu0_ac_fea=NULL;}
  if (feamatrix)  {delete feamatrix; feamatrix=NULL;}
}


RNNLM::RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1,
             const vector<int> &lsizes, int fvocsize,
             bool bformat, int debuglevel): inmodelfile(inmodelfile_1),
             inputwlist(inputwlist_1), outputwlist(outputwlist_1),
             layersizes(lsizes), debug(debuglevel), binformat(bformat) {
  // now we only support 1 hidden layer
  assert(lsizes.size() == 3);
  init();
  LoadRNNLM(inmodelfile);
  ReadWordlist(inputwlist, outputwlist);

  setFullVocsize (fvocsize);

  resetAc = new float[layersizes[1]];
  memcpy(resetAc, neu0_ac_hist->gethostdataptr(), sizeof(float)*layersizes[1]);
}

void RNNLM::copyToHiddenLayer(const vector<float> &hidden) {
  const float *srcac;
  float *dstac;
  assert(hidden.size() == layersizes[1]);
  srcac = hidden.data(); // TODO
  dstac = neu_ac[1]->gethostdataptr();
//  memcpy (dstac, srcac, sizeof(float)*layersizes[1]);
  std::copy(hidden.begin(), hidden.end(), dstac); // TODO
}

void RNNLM::fetchHiddenLayer(vector<float> *context_out) {
  if (context_out == NULL) {
    return;
  }
  const float *srcac;
  float *dstac;
  assert(context_out->size() == layersizes[1]);
  srcac = neu_ac[1]->gethostdataptr();
  dstac = context_out->data(); // TODO(hxu)
//  memcpy (dstac, srcac, sizeof(float)*layersizes[1]);
  std::copy(neu_ac[1]->gethostdataptr(),
            neu_ac[1]->gethostdataptr() + layersizes[1], dstac); // TODO
}

float RNNLM::computeConditionalLogprob(int current_word,
                                       const vector<int> &history_words,
                                       const vector<float> &context_in,
                                       vector<float> *context_out) {
  float ans = 0.0;
  copyToHiddenLayer(context_in);
  int last_word = 0; // <s>
//  if (history_words.size() != 0) {
    last_word = history_words[history_words.size() - 1];
//  }
  ans = forward(last_word, current_word);
  fetchHiddenLayer(context_out);

  return ans;
}

// allocate memory for RNNLM model
void RNNLM::allocMem (vector<int> &layersizes)
{
  int i;
  num_layer = layersizes.size() - 1;
  if (num_layer < 2) {
    printf ("ERROR: the number of layers (%d) should not be smaller than 2\n", num_layer);
  }
  inputlayersize = layersizes[0] + layersizes[1];
  outputlayersize = layersizes[num_layer];
  layer0_hist = new matrix (layersizes[1], layersizes[1]); // TODO not sure what this is for
  neu0_ac_hist = new matrix (layersizes[1], minibatch);
  layers.resize(num_layer);
  neu_ac.resize(num_layer+1);
  for (i = 0; i < num_layer; i++) {
    int nrow = layersizes[i];
    int ncol = layersizes[i + 1];
    layers[i] = new matrix(nrow, ncol);
  }
  for (i = 0; i < layersizes.size(); i++) {
    int nrow = layersizes[i];
    neu_ac[i] = new matrix(nrow, minibatch);
  }
  if (dim_fea > 0) {
    layer0_fea = new matrix(dim_fea, layersizes[1]);
    neu0_ac_fea = new matrix(dim_fea, minibatch);
    neu0_ac_fea->initmatrix();
  }
  if (nclass > 0) {
    layerN_class = new matrix(layersizes[num_layer-1], nclass);
    neuN_ac_class = new matrix(nclass, minibatch);
    neuN_ac_class->initmatrix();
  }
}

void RNNLM::printPPLInfo() {
  string str;
  printf ("model file :       %s\n", inmodelfile.c_str());
  printf ("input  list:       %s\n", inputwlist.c_str());
  printf ("output list:       %s\n", outputwlist.c_str());
  printf ("num   layer:       %d\n", num_layer);
  for (int i=0; i<=num_layer; i++) {
    printf ("#layer[%d]  :       %d\n", i, layersizes[i]);
  }
  printf ("independent:       %d\n", independent);
  printf ("test file  :       %s\n", testfile.c_str());
  printf ("nglm file  :       %s\n", nglmstfile.c_str());
  printf ("lambda (rnn):      %f\n", lambda);
  printf ("fullvocsize:       %d\n", fullvocsize);
  printf ("debug level:       %d\n", debug);
  printf ("nthread    :       %d\n", nthread);
  if (nodetype == 0) 
    str = "sigmoid";
  else if (nodetype == 1) 
    str = "relu";
  else {
    printf ("ERROR: unknown type of hidden node: %d\n", nodetype); exit (0);
  }
  printf ("node type  :       %s\n", str.c_str());
  if (nodetype == 1) {
    printf ("relu ratio :       %f\n", reluratio);
  }
}

bool RNNLM::calppl(string testfilename, float intpltwght, string nglmfile) {
  int i, wordcn, nwordoov, cnt;
  vector<string> linevec;
  FILEPTR fileptr;
  float prob_rnn, prob_ng, prob_int, logp_rnn,
        logp_ng, logp_int, ppl_rnn, ppl_ng,
        ppl_int;
  bool flag_intplt = false, flag_oov = false;
  FILE *fptr_nglm=NULL;
  auto_timer timer;
  timer.start();
  string word;
  testfile = testfilename;
  nglmstfile = nglmfile;
  lambda = intpltwght;
  if (debug > 1) {
    printPPLInfo ();
  }

  if (!nglmfile.empty()) {
    fptr_nglm = fopen(nglmfile.c_str(), "r");
    if (fptr_nglm == NULL) {
      printf ("ERROR: Failed to open ng stream file: %s\n", nglmfile.c_str());
      exit (0);
    }
    flag_intplt = true;
  }
  fileptr.open(testfile);

  wordcn = 0;
  nwordoov = 0;
  logp_int = 0;
  logp_rnn = 0;
  logp_ng = 0;

  for (int i=0; i<layersizes[1]; i++)
    neu0_ac_hist->assignhostvalue(i, 0, RESETVALUE);

  if (debug > 1) {
    if (flag_intplt) {
      printf("\nId\tP_rnn\t\tP_ng\t\tP_int\t\tWord\n");
    }
    else {
      printf("\nId\tP_rnn\t\tWord\n");
    }
  }

  while (!fileptr.eof()) {
    if (dim_fea > 0) {
      int feaid = fileptr.readint();
      assert (feaid >=0 && feaid < num_fea);
      float *feaptr = feamatrix->gethostdataptr(0, feaid);
      float *acptr  = neu0_ac_fea->gethostdataptr();
      memcpy(acptr, feaptr, sizeof(float)*dim_fea);
    }
    fileptr.readline(linevec, cnt);
    float sentence_loglike = 0.0;

    if (linevec.size() > 0) {
      if (linevec[cnt-1] != "</s>") {
        linevec.push_back("</s>");
        cnt ++;
      }

      assert (cnt == linevec.size());

      if (linevec[0] == "<s>")
        i = 1;
      else
        i = 0;

      prevword = inStartindex;
      if (independent) {
        ResetRechist(); /// TODO
      }

      for (; i < cnt; i++) {
        word = linevec[i];
        if (outputmap.find(word) == outputmap.end()) {
          curword = outOOSindex;
        }
        else {
          curword = outputmap[word];
        }
        prob_rnn = forward(prevword, curword);

        if (curword == outOOSindex)
          // uniformly distribute the probability mass among OOS's
          prob_rnn /= (fullvocsize - layersizes[num_layer] + 1);

        copyRecurrentAc ();

        if (flag_intplt) {
          if (fscanf (fptr_nglm, "%f\n", &prob_ng) != 1) {
            printf ("ERROR: Failed to read ngram prob from ng stream file!\n");
            exit (0);
          }
          if (fabs(prob_ng) < 1e-9)   {flag_oov = true;}
          else                        flag_oov = false;
        }
        prob_int = lambda * prob_rnn + (1 - lambda) * prob_ng;

        if (inputmap.find(word) == inputmap.end()) {
          prevword = inOOSindex;
        }
        else {
          prevword = inputmap[word];
        }

        if (!flag_oov) {
          logp_rnn += log10(prob_rnn);
          logp_ng  += log10(prob_ng);
          logp_int += log10(prob_int);
        }
        else {
          nwordoov++;
        }

        wordcn++;

        if (debug > 1) {
          if (flag_intplt)
            printf ("%d\t%.10f\t%.10f\t%.10f\t%s", curword, prob_rnn, prob_ng,
                                                   prob_int, word.c_str());
          else
            printf ("%d\t%.10f\t%s", curword, prob_rnn, word.c_str());

          if (curword == outOOSindex) {
            if (flag_oov)   printf ("<OOV>");
            else            printf ("<OOS>");
          }
          printf ("\n");
        }

        if (debug > 2) {
          if (wordcn % 10000 == 0) {
            float nwordspersec = wordcn / (timer.stop());
            printf ("eval speed  %.4f Words/sec\n", nwordspersec);
          }
        }
        sentence_loglike += log10(prob_rnn);
      }
    }

    if (cnt != 0)
      printf ("per-sentence log-likelihood: %f\n", sentence_loglike);
  }

  if (debug > 2) {
    float nwordspersec = wordcn / (timer.stop());
    printf ("eval speed  %.4f Words/sec\n", nwordspersec);
  }

  ppl_rnn = exp10(-logp_rnn/(wordcn-nwordoov));
  ppl_ng  = exp10(-logp_ng/(wordcn-nwordoov));
  ppl_int = exp10(-logp_int/(wordcn-nwordoov));

  if (flag_intplt) {
    printf ("Total word: %d\tOOV word: %d\n", wordcn, nwordoov);
    printf ("N-Gram log probability: %.3f\n", logp_ng);
    printf ("RNNLM  log probability: %.3f\n", logp_rnn);
    printf ("Intplt log probability: %.3f\n\n", logp_int);
    printf ("N-Gram PPL : %.3f\n", ppl_ng);
    printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
    printf ("Intplt PPL : %.3f\n", ppl_int);
  }
  else {
    printf ("Total word: %d\tOOV word: %d\n", wordcn, nwordoov);
    printf ("Average logp: %f\n", logp_rnn/log10(2)/wordcn);
    printf ("RNNLM  log probability: %.3f\n", logp_rnn);
    printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
  }
  fileptr.close();

  if (fptr_nglm) {
    fclose(fptr_nglm);
  }

  return SUCCESS;
}

bool RNNLM::calnbest (string testfilename, float intpltwght, string nglmfile) {
    int i, wordcn, cnt, nbestid, prevnbestid=-1, sentcnt=0, nword;
    vector<string> linevec, maxlinevec;
    FILEPTR fileptr;
    float prob_rnn, prob_ng, prob_int, logp_rnn,
          logp_ng, logp_int, ppl_rnn, ppl_ng,
          ppl_int, sentlogp, acscore, lmscore, score, maxscore;
    bool flag_intplt = false;
    FILE *fptr_nglm=NULL;
    auto_timer timer;
    timer.start();
    string word;
    testfile = testfilename;
    nglmstfile = nglmfile;
    lambda = intpltwght;
    if (debug > 1)
    {
        printPPLInfo ();
    }
    if (!nglmfile.empty())
    {
        fptr_nglm = fopen (nglmfile.c_str(), "r");
        if (fptr_nglm == NULL)
        {
            printf ("ERROR: Failed to open ng stream file: %s\n", nglmfile.c_str());
            exit (0);
        }
        flag_intplt = true;
    }
    fileptr.open(testfile);

    wordcn = 0;
    logp_int = 0;
    logp_rnn = 0;
    logp_ng = 0;
    for (int i=0; i<layersizes[1]; i++)
    {
        neu0_ac_hist->assignhostvalue(i, 0, RESETVALUE);
        resetAc[i] = RESETVALUE;
    }
    while (!fileptr.eof())
    {
        if (dim_fea > 0)
        {
            int feaid = fileptr.readint();
            assert (feaid >=0 && feaid < num_fea);
            float *feaptr = feamatrix->gethostdataptr(0, feaid);
            float *acptr  = neu0_ac_fea->gethostdataptr();
            memcpy(acptr, feaptr, sizeof(float)*dim_fea);
        }
        fileptr.readline(linevec, cnt);
        if (linevec.size() > 0)
        {
            if (linevec[cnt-1] != "</s>")
            {
                linevec.push_back("</s>");
                cnt ++;
            }
            assert (cnt == linevec.size());
            // the first two iterms for linevec are: <s> nbestid
            // nbid acscore  lmscore  nword <s> ... </s>
            // 0    2750.14 -6.03843    2   <s> HERE YEAH </s>
            // erase the first <s> and last</s>
            vector<string>::iterator it= linevec.begin();
            linevec.erase(it);
            cnt --;
            // it = linevec.end();
            // it --;
            // linevec.erase(it);
            nbestid = string2int(linevec[0]);

            if (nbestid != prevnbestid)
            {
                memcpy(resetAc, neu0_ac_hist->gethostdataptr(), sizeof(float)*layersizes[1]);
                if (prevnbestid != -1)
                {
                    for (i=4; i<maxlinevec.size(); i++)
                    {
                        word = maxlinevec[i];
                        if (word != "<s>" && word != "</s>")
                        {
                            printf (" %s", word.c_str());
                        }
                    }
                    printf ("\n");
                }
                maxscore = -10000.0;
            }
            else
            {
                memcpy(neu0_ac_hist->gethostdataptr(), resetAc, sizeof(float)*layersizes[1]);
            }

            acscore = string2float(linevec[1]);
            lmscore = string2float(linevec[2]);
            assert(lmscore == lmscore); // TODO
            nword   = string2int(linevec[3]);
            if (linevec[4] == "<s>")    i = 5;
            else                        i = 4;
            sentlogp = 0;
            prevword = inStartindex;
            if (independent)
            {
                ResetRechist();
            }
            for (; i<cnt; i++)
            {
                word = linevec[i];
                if (outputmap.find(word) == outputmap.end())
                {
                    curword = outOOSindex;
                }
                else
                {
                    curword = outputmap[word];
                }
                prob_rnn = forward (prevword, curword);
                if (curword == outOOSindex)     prob_rnn /= (fullvocsize-layersizes[num_layer]+1);
                copyRecurrentAc ();

                if (flag_intplt)
                {
                    if (fscanf (fptr_nglm, "%f\n", &prob_ng) != 1)
                    {
                        printf ("ERROR: Failed to read ngram prob from ng stream file!\n");
                        exit (0);
                    }
                }
                prob_int = lambda*prob_rnn + (1-lambda)*prob_ng;
                if (inputmap.find(word) == inputmap.end())
                {
                    prevword = inOOSindex;
                }
                else
                {
                    prevword = inputmap[word];
                }
                logp_rnn += log10(prob_rnn);
                logp_ng  += log10(prob_ng);
                logp_int += log10(prob_int);
                sentlogp += log10(prob_int);
                wordcn ++;
                if (debug == 1)
                {
                    printf ("%f ", log10(prob_int));
                }
                if (debug > 1)
                {
                    if (flag_intplt)
                        printf ("%d\t%.10f\t%.10f\t%.10f\t%s", curword, prob_rnn, prob_ng, prob_int, word.c_str());
                    else
                        printf ("%d\t%.10f\t%s", curword, prob_rnn, word.c_str());
                    if (curword == outOOSindex)
                    {
                        printf ("<OOS>");
                    }
                    printf ("\n");
                }
                if (debug > 1)
                {
                    if (wordcn % 10000 == 0)
                    {
                        float nwordspersec = wordcn / (timer.stop());
                        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
                    }
                }
            }
            sentcnt ++;
            if (debug == 1)
            {
                printf ("sent=%f %d\n", sentlogp, sentcnt);
            }
            if (debug == 0)
            {
                score = acscore + sentlogp*lmscale + ip*(nword+1);
                if (score > maxscore)
                {
                    maxscore = score;
                    maxlinevec = linevec;
                }
                for (i=4; i<cnt; i++)
                {
                    word = linevec[i];
                    // printf (" %s", word.c_str());
                }
                // printf ("\n");
            }
            fflush(stdout);
            prevnbestid = nbestid;
        }
    }
    for (i=4; i<maxlinevec.size(); i++)
    {
        word = maxlinevec[i];
        if (word != "<s>" && word != "</s>")
        {
            printf (" %s", word.c_str());
        }
    }
    printf ("\n");
    if (debug > 1)
    {
        float nwordspersec = wordcn / (timer.stop());
        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
    }
    ppl_rnn = exp10(-logp_rnn/(wordcn));
    ppl_ng  = exp10(-logp_ng/(wordcn));
    ppl_int = exp10(-logp_int/(wordcn));
    if (debug > 1)
    {
        if (flag_intplt)
        {
            printf ("Total word: %d\n", wordcn);
            printf ("N-Gram log probability: %.3f\n", logp_ng);
            printf ("RNNLM  log probability: %.3f\n", logp_rnn);
            printf ("Intplt log probability: %.3f\n\n", logp_int);
            printf ("N-Gram PPL : %.3f\n", ppl_ng);
            printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
            printf ("Intplt PPL : %.3f\n", ppl_int);
        }
        else
        {
            printf ("Total word: %d\n", wordcn);
            printf ("Average logp: %f\n", logp_rnn/log10(2)/wordcn);
            printf ("RNNLM  log probability: %.3f\n", logp_rnn);
            printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
        }
    }
    fileptr.close();

    if (fptr_nglm)
    {
        fclose(fptr_nglm);
    }
    return SUCCESS;
}

void RNNLM::InitVariables() {
  int i, j;
  counter = 0;
  logp = 0;
  wordcn = 0;
  if (independent) {
    for (i = 0; i < neu0_ac_hist->rows(); i++)
      for (j = 0; j < neu0_ac_hist->cols(); j++)
        neu0_ac_hist->assignhostvalue(i, j, RESETVALUE);
  }
  if (traincritmode == 1 || traincritmode == 2) {
    lognorm_mean = 0;
    lognorm_var  = 0;
  }
}

void RNNLM::LoadRNNLM(string modelname) {
  if (binformat) {
    LoadBinaryRNNLM(modelname);
  }
  else {
    LoadTextRNNLM(modelname);
  }
}

void RNNLM::LoadTextRNNLM(string modelname) {
  int i, a, b;
  float v;
  char word[1024];
  FILE *fptr = NULL;
  // read model file
  fptr = fopen (modelname.c_str(), "r");
  if (fptr == NULL) {
    printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelname.c_str());
    exit (0);
  }
  fscanf (fptr, "cuedrnnlm v%f\n", &v);
  if (v != version) {
    printf ("Error: the version of rnnlm model(v%.1f) is not consistent with binary supported(v%.1f)\n", v, version);
    exit (0);
  }

  fscanf (fptr, "train file: %s\n", word);     trainfile = word;
  fscanf (fptr, "valid file: %s\n", word);     validfile = word;
  fscanf (fptr, "number of iteration: %d\n", &iter);
  fscanf (fptr, "#train words: %d\n", &trainwordcnt);
  fscanf (fptr, "#valid words: %d\n", &validwordcnt);
  fscanf (fptr, "#layer: %d\n", &num_layer);
  layersizes.resize(num_layer+1);
  for (i=0; i<layersizes.size(); i++) {
    fscanf (fptr, "layer %d size: %d\n", &b, &a);
    assert(b==i);
    layersizes[i] = a;
  }

  fscanf (fptr, "feature dimension: %d\n", &dim_fea);
  fscanf (fptr, "class layer dimension: %d\n", &nclass);
  fscanf (fptr, "fullvoc size: %d\n", &fullvocsize);
  allocMem (layersizes);

  fscanf (fptr, "independent mode: %d\n", &independent);
  fscanf (fptr, "train crit mode: %d\n",  &traincritmode);
  fscanf (fptr, "log norm: %f\n", &lognormconst);
  fscanf (fptr, "hidden node type: %d\n", &nodetype);

  if (nodetype == 1) {
    fscanf (fptr, "relu ratio: %f\n", &reluratio);
  }

  for (i = 0; i < num_layer; i++) {
    fscanf (fptr, "layer %d -> %d\n", &a, &b);
    assert (a==i);
    assert (b==(i+1));
    for (a=0; a<layersizes[i]; a++) {
      for (b=0; b<layersizes[i+1]; b++) {
        fscanf (fptr, "%f", &v);
        layers[i]->assignhostvalue(a, b, v);
      }
      fscanf (fptr, "\n");
    }
  }

  fscanf (fptr, "recurrent layer 1 -> 1\n");
  for (a=0; a<layersizes[1]; a++) {
    for (b=0; b<layersizes[1]; b++) {
      fscanf (fptr, "%f", &v);
      layer0_hist->assignhostvalue(a, b, v);
    }
    fscanf (fptr, "\n");
  }

  if (dim_fea > 0) {
    fscanf (fptr, "feature layer weight\n");
    for (a = 0; a < dim_fea; a++) {
      for (b = 0; b < layersizes[1]; b++) {
        fscanf (fptr, "%f", &v);
        layer0_fea->assignhostvalue(a, b, v);
      }
      fscanf (fptr, "\n");
    }
  }

  if (nclass > 0) {
    fscanf (fptr, "class layer weight\n");
    for (a=0; a<layersizes[num_layer-1]; a++) {
      for (b=0; b<nclass; b++) {
        fscanf (fptr, "%f", &v);
        layerN_class->assignhostvalue(a, b, v);
      }
      fscanf (fptr, "\n");
    }
  }
  fscanf (fptr, "hidden layer ac\n");

  for (a=0; a<layersizes[1]; a++) {
    fscanf (fptr, "%f", &v);
    for (b=0; b<minibatch; b++)
      neu0_ac_hist->assignhostvalue(a, b, v);
  }

  fscanf (fptr, "\n");
  fscanf (fptr, "%d", &a);
  if (a != CHECKNUM) {
    printf ("ERROR: failed to read the check number(%d) when reading model\n", CHECKNUM);
    exit (0);
  }
  if (debug > 1) {
    printf ("Successfully loaded model: %s\n", modelname.c_str());
  }
  fclose (fptr);
}

void RNNLM::LoadBinaryRNNLM(string modelname)
{
  int i, a, b;
  float v;
//    char word[1024];
  FILE *fptr = NULL;
  fptr = fopen (modelname.c_str(), "rb");
  if (fptr == NULL)
  {
      printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelname.c_str());
      exit (0);
  }
  fread (&v, sizeof(float), 1, fptr);
  if (v != version)
  {
      printf ("Error: the version of rnnlm model(v%.1f) is not consistent with binary supported(v%.1f)\n", v, version);
      exit (0);
  }
  fread (&iter, sizeof(int), 1, fptr);
  fread (&num_layer, sizeof(int), 1, fptr);
  layersizes.resize(num_layer+1);
  for (i=0; i<layersizes.size(); i++)
  {
      fread (&a, sizeof(int), 1, fptr);
      layersizes[i] = a;
  }
  fread (&dim_fea, sizeof(int), 1, fptr);
  fread (&nclass, sizeof(int), 1, fptr);
  fread (&fullvocsize, sizeof(int), 1, fptr);
  allocMem (layersizes);

  fread (&independent, sizeof(int), 1, fptr);
  fread (&traincritmode, sizeof(int), 1, fptr);
  fread (&nodetype, sizeof(int), 1, fptr);
  if (nodetype == 1) {
    fwrite (&reluratio, sizeof(float), 1, fptr);
  }

  for (i=0; i<num_layer; i++) {
    for (a=0; a<layersizes[i]; a++) {
      for (b=0; b<layersizes[i+1]; b++) {
        fread (&v, sizeof(float), 1, fptr);
        layers[i]->assignhostvalue(a, b, v);
      }
    }
  }
  for (a=0; a<layersizes[1]; a++) {
    for (b=0; b<layersizes[1]; b++) {
      fread (&v, sizeof(float), 1, fptr);
      layer0_hist->assignhostvalue(a, b, v);
    }
  }

  if (dim_fea > 0) {
    for (a=0; a<dim_fea; a++) {
      for (b=0; b<layersizes[1]; b++) {
        fread (&v, sizeof(float), 1, fptr);
        layer0_fea->assignhostvalue(a, b, v);
      }
    }
  }

  if (nclass > 0) {
    for (a=0; a<layersizes[num_layer-1]; a++) {
      for (b=0; b<nclass; b++) {
        fread (&v, sizeof(float), 1, fptr);
        layerN_class->assignhostvalue(a, b, v);
      }
    }
  }

  for (a=0; a<layersizes[1]; a++) {
    fread (&v, sizeof(float), 1, fptr);
    neu_ac[1]->assignhostvalue(a, 0, v);
  }

  fread (&a, sizeof(int), 1, fptr);

  if (a != CHECKNUM) {
    printf ("ERROR: failed to read the check number(%d) when reading model\n", CHECKNUM);
    exit (0);
  }

  if (debug > 0) {
    printf ("Successfully loaded model: %s\n", modelname.c_str());
  }
  fclose (fptr);
}

// read intput and output word list
void RNNLM::ReadWordlist (string inputlist, string outputlist) {
    //index 0 for <s> and </s> in input and output layer
    //last node for <OOS>
  int i;
//    float v;
  char word[1024];
  FILE *finlst, *foutlst;
  finlst = fopen (inputlist.c_str(), "r");
  foutlst = fopen (outputlist.c_str(), "r");
  if (finlst == NULL || foutlst == NULL) {
    printf ("ERROR: Failed to open input (%s) or output list file(%s)\n",
              inputlist.c_str(), outputlist.c_str());
    exit (0);
  }

  inputmap.insert(make_pair(string("<s>"), 0));
  outputmap.insert(make_pair(string("</s>"), 0));
  inputvec.clear();
  outputvec.clear();
  inputvec.push_back("<s>");
  outputvec.push_back("</s>");
  int index = 1;

  while (!feof(finlst)) {
    if(fscanf (finlst, "%d%s", &i, word) == 2) {
      if (inputmap.find(word) == inputmap.end()) {
        inputmap[word] = index;
        inputvec.push_back(word);
        index ++;
      }
    }
  }

  if (inputmap.find("<OOS>") == inputmap.end()) {
    inputmap.insert(make_pair(string("<OOS>"), index));
    inputvec.push_back("<OOS>");
  }
  else {
    assert (inputmap["<OOS>"] == inputvec.size()-1);
  }

  index = 1;
  // allocate memory for class information
  if (nclass > 0) {
    word2class = new int[layersizes[num_layer]];
    classinfo = new int[nclass*3];
    classinfo[0] = 0;
  }
  int clsid, prevclsid = 0;

  while (!feof(foutlst)) {
    if (nclass > 0) {
      if (fscanf(foutlst, "%d%s%d", &i, word, &clsid) == 3) {
        if (outputmap.find(word) == outputmap.end())
        {
          outputmap[word] = index;
          outputvec.push_back(word);
          index ++;
        }
        int idx = outputmap[word];
        word2class[idx] = clsid;
        if (clsid != prevclsid)
        {
          classinfo[prevclsid*3+1] = idx-1;
          classinfo[prevclsid*3+2] = idx-classinfo[prevclsid*3];
          classinfo[3*clsid]=idx;
        }
        prevclsid = clsid;
      }
    }
    else
    {
      if (fscanf(foutlst, "%d%s", &i, word) == 2)
      {
        if (outputmap.find(word) == outputmap.end())
        {
          outputmap[word] = index;
          outputvec.push_back(word);
          index ++;
        }
      }
    }
  }

  if (nclass > 0) {
    classinfo[prevclsid*3+1] = layersizes[num_layer]-1;
    classinfo[prevclsid*3+2] = layersizes[num_layer]-classinfo[prevclsid*3];
  }
  if (outputmap.find("<OOS>") == outputmap.end()) {
    outputmap.insert(make_pair(string("<OOS>"), index));
    outputvec.push_back("<OOS>");
  }
  else {
    assert (outputmap["<OOS>"] == outputvec.size()-1);
  }
  assert (inputvec.size() == layersizes[0]);
  assert (outputvec.size() == layersizes[num_layer]);
  inStartindex = 0;
  outEndindex  = 0;
  inOOSindex   = inputvec.size() - 1;
  outOOSindex  = outputvec.size() - 1;
  assert (outOOSindex == outputmap["<OOS>"]);
  assert (inOOSindex == inputmap["<OOS>"]);
  fclose (finlst);
  fclose (foutlst);
}

void RNNLM::setFullVocsize (int n)
{
  if (n == 0) {
    fullvocsize = layersizes[0];
  }
  else {
    fullvocsize = n;
  }
}

void RNNLM::copyRecurrentAc ()
{
  float *srcac, *dstac;
  assert (minibatch == 1);
  srcac = neu_ac[1]->gethostdataptr();
  dstac = neu0_ac_hist->gethostdataptr();
  memcpy (dstac, srcac, sizeof(float) * layersizes[1]);
}

void RNNLM::ResetRechist() {
  for (int i=0; i<layersizes[1]; i++) {
    neu0_ac_hist->assignhostvalue(i, 0, RESETVALUE);
    resetAc[i] = RESETVALUE;
  }
}

float RNNLM::forward (int prevword, int curword) {
  int a, b, nrow, ncol;
  nrow = layersizes[1];
  ncol = layersizes[1];
  float *srcac, *wgt, *dstac;

   // neu0 -> neu1
  for (a = 0; a < layers.size(); a++) {
    if (a==0) {
      srcac = neu0_ac_hist->gethostdataptr();
      wgt   = layer0_hist->gethostdataptr();
      dstac = neu_ac[1]->gethostdataptr();
      nrow  = layersizes[1];
      ncol  = layersizes[1];
      memset(dstac, 0, sizeof(float) * ncol);

      for (b = 0; b < ncol; b++) {
        dstac[b] = layers[0]->fetchhostvalue(prevword, b);
      }

      if (dim_fea > 0) {
        float *feasrcac = neu0_ac_fea->gethostdataptr();
        float *feawgt   = layer0_fea->gethostdataptr();
        matrixXvector (feasrcac, feawgt, dstac, dim_fea, ncol);
      }
    }
    else {
      srcac = neu_ac[a]->gethostdataptr();
      wgt   = layers[a]->gethostdataptr();
      dstac = neu_ac[a + 1]->gethostdataptr();
      nrow  = layersizes[a];
      ncol  = layersizes[a + 1];
      memset(dstac, 0, sizeof(float) * ncol);
    }

    if (a + 1== num_layer) {
      if (lognormconst < 0) {
        if (nclass > 0) {
          float *clsdstac = neuN_ac_class->gethostdataptr();
          float *clswgt = layerN_class->gethostdataptr();
          int ncol_cls = nclass;
          matrixXvector (srcac, clswgt, clsdstac, nrow, ncol_cls);
          neuN_ac_class->hostsoftmax();
          int clsid = word2class[curword];
          int swordid = classinfo[clsid*3];
          int ewordid = classinfo[clsid*3+1];
          int nword   = classinfo[clsid*3+2];
          matrixXvector(srcac, wgt + swordid * nrow, dstac + swordid, nrow, nword);
          neu_ac[a+1]->hostpartsoftmax(swordid, ewordid);
        }
        else {
          matrixXvector (srcac, wgt, dstac, nrow, ncol);
          neu_ac[a+1]->hostsoftmax();
        }
      }
      else {
        float v = 0;
        for (int i = 0; i < nrow; i++) {
          // v += srcac[i]*wgt[i+curword*nrow];
          v += neu_ac[a]->fetchhostvalue(i, 0) * layers[a]->fetchhostvalue(i, curword);
        }
        dstac[curword] = exp(v-lognormconst);
      }
    }
    else {
      matrixXvector (srcac, wgt, dstac, nrow, ncol);
      if (nodetype == 0) {
        neu_ac[a+1]->hostsigmoid();
      }
      else if (nodetype == 1) {
        neu_ac[a+1]->hostrelu(reluratio);
      }
      else {
        printf ("ERROR: unknow type of hidden node: %d\n", nodetype);
        exit (0);
      }
    }
  }
  if (nclass > 0) {
    int clsid = word2class[curword];
    return neu_ac[num_layer]->fetchhostvalue(curword, 0)
            * neuN_ac_class->fetchhostvalue(clsid, 0);
  }
  else {
    return neu_ac[num_layer]->fetchhostvalue(curword, 0);
  }
}

void RNNLM::matrixXvector (float *src, float *wgt, float *dst, int nr, int nc) {
  int i, j;
#if 1
#ifdef NUM_THREAD
#pragma omp parallel for num_threads(NUM_THREAD)
#else
#pragma omp parallel for num_threads(nthread)
#endif
#endif
  for (i = 0; i < nc; i++) {
    for (j = 0; j < nr; j++) {
      dst[i] += src[j] * wgt[j + i * nr];
    }
  }
  return;
}

void RNNLM::ReadFeaFile(string filestr) {
  int i, j, t;
  float value;
  FILE *fptr = fopen (filestr.c_str(), "r");
  if (fptr == NULL) {
    printf ("Error: Failed to open feature file: %s\n", filestr.c_str());
    exit(0);
  }
  fscanf (fptr, "%d %d", &num_fea, &dim_fea);
  // if the fea file is two large, just allocate cpu memory
  feamatrix = new matrix (dim_fea, num_fea);
  feamatrix->initmatrix();

  if (debug > 1) {
      printf ("%d lines feature (with %d dimensions) will be read from %s\n",
              num_fea, dim_fea, filestr.c_str());
  }
  i = 0;
  while (i < num_fea) {
    if (feof(fptr))         break;
    fscanf (fptr, "%d", &j);
    assert (j == i);
    for (t=0; t<dim_fea; t++) {
      fscanf (fptr, "%f", &value);
      feamatrix->assignhostvalue(t, i, value);
    }
    i++;
  }

  if (i != num_fea) {
    printf ("Warning: only read %d lines from the feature file: %s, should be %d lines\n",
             i, filestr.c_str(), num_fea);
  }
  if (debug > 1) {
    printf ("%d feature lines (with %d dimensions) is read from %s successfully\n",
             num_fea, dim_fea,  filestr.c_str());
  }
  fclose(fptr);

  // allocate memeory for additional feature in input layer
  // if memory already allocated during laoding model
  if (layer0_fea == NULL) {
    layer0_fea = new matrix(dim_fea, layersizes[1]);
    layer0_fea->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    neu0_ac_fea = new matrix (dim_fea, minibatch);
    neu0_ac_fea->initmatrix();
  }
}


} // namespace cued_rnnlm
