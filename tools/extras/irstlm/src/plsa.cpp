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


#include <iostream>
#include "cmd.h"
#include <pthread.h>
#include "thpool.h"
#include "util.h"
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "ngramtable.h"
#include "doc.h"
#include "cplsa.h"

using namespace std;
using namespace irstlm;

void print_help(int TypeFlag=0){
    std::cerr << std::endl << "plsa -  probabilistic latent semantic analysis modeling" << std::endl;
    std::cerr << std::endl << "USAGE:"  << std::endl;
    std::cerr << "       plsa -tr|te=<text>  -m=<model> -t=<n> [options]" << std::endl;
    std::cerr << std::endl << "DESCRIPTION:" << std::endl;
    std::cerr << "       Train a PLSA model from a corpus and test it to infer topic or word " << std::endl;
    std::cerr << "       distributions from other texts." << std::endl;
    std::cerr << "       Notice: multithreading is available both for training and inference." << std::endl;
    
    std::cerr << std::endl << "OPTIONS:" << std::endl;
    
    
    FullPrintParams(TypeFlag, 0, 1, stderr);
    
    std::cerr << std::endl << "EXAMPLES:" << std::endl;
    std::cerr <<"       (1) plsa -tr=<text> -t=<n> -m=<model> " << std::endl;
    std::cerr <<"           Train a PLSA model <model> with <n> topics on text <text> " << std::endl;
    std::cerr <<"           Example of <text> content:" << std::endl;
    std::cerr <<"           3" << std::endl;
    std::cerr <<"           <d> hello world ! </d>" << std::endl;
    std::cerr <<"           <d> good morning good afternoon </d>" << std::endl;
    std::cerr <<"           <d> welcome aboard </d>" << std::endl;
    std::cerr <<"       (2) plsa -m=<model> -te=<text> -tf=<features>" << std::endl;
    std::cerr <<"           Infer topic distribution with model <model> for each doc in <text>" << std::endl;
    std::cerr <<"       (3) plsa -m=<model> -te=<text> -wf=<features>" << std::endl;
    std::cerr <<"           Infer word distribution with model <model> for each doc in <text>" << std::endl;
    std::cerr << std::endl;
}

void usage(const char *msg = 0)
{
  if (msg){
    std::cerr << msg << std::endl;
	}
  else{
		print_help();
	}
}

int main(int argc, char **argv){
    char *dictfile=NULL;
    char *trainfile=NULL;
    char *testfile=NULL;
    char *topicfeaturefile=NULL;
    char *wordfeaturefile=NULL;
    char *modelfile=NULL;
    char *tmpdir = getenv("TMP");
    char *txtfile=NULL;
    bool forcemodel=false;
    
    int topics=0;              //number of topics
    int specialtopic=0;       //special topic: first st dict words
    int iterations=10;       //number of EM iterations to run
    int threads=1;           //current EM iteration for multi-thread training
    bool help=false;
    bool memorymap=true;
    int prunethreshold=3;
    int topwords=20;
    DeclareParams((char*)
                  
                  "Train", CMDSTRINGTYPE|CMDMSG, &trainfile, "<fname> : training  text collection ",
                  "tr", CMDSTRINGTYPE|CMDMSG, &trainfile, "<fname> : training text collection ",
                  
                  "Model", CMDSTRINGTYPE|CMDMSG, &modelfile, "<fname> : model file",
                  "m", CMDSTRINGTYPE|CMDMSG, &modelfile, "<fname> : model file",
                  
                  "TopWordsFile", CMDSTRINGTYPE|CMDMSG, &txtfile, "<fname> to write top words per topic",
                  "twf", CMDSTRINGTYPE|CMDMSG, &txtfile, "<fname> to write top words per topic",
               
                  "PruneFreq", CMDINTTYPE|CMDMSG, &prunethreshold, "<count>: prune words with freq <= count (default 3)",
                  "pf", CMDINTTYPE|CMDMSG, &prunethreshold, "<count>: <count>: prune words with freq <= count (default 3)",
                  
                  "TopWordsNum", CMDINTTYPE|CMDMSG, &topwords, "<count>: number of top words per topic ",
                  "twn", CMDINTTYPE|CMDMSG, &topwords, "<count>: number of top words per topic",
                
                  "Test", CMDSTRINGTYPE|CMDMSG, &testfile, "<fname> : inference text collection file",
                  "te", CMDSTRINGTYPE|CMDMSG, &testfile, "<fname> : inference text collection file",
                  
                  "WordFeatures", CMDSTRINGTYPE|CMDMSG, &wordfeaturefile, "<fname> : unigram feature file",
                  "wf", CMDSTRINGTYPE|CMDMSG, &wordfeaturefile,"<fname> : unigram feature file",
                  
                  "TopicFeatures", CMDSTRINGTYPE|CMDMSG, &topicfeaturefile, "<fname> : topic feature file",
                  "tf", CMDSTRINGTYPE|CMDMSG, &topicfeaturefile, "<fname> : topic feature file",
                  
                  "Topics", CMDINTTYPE|CMDMSG, &topics, "<count> : number of topics (default 0)",
                  "t", CMDINTTYPE|CMDMSG, &topics,"<count> : number of topics (default 0)",
                  
                  "SpecialTopic", CMDINTTYPE|CMDMSG, &specialtopic, "<count> : put top-<count> frequent words in a special topic (default 0)",
                  "st", CMDINTTYPE|CMDMSG, &specialtopic, "<count> :  put top-<count> frequent words in a special topic (default 0)",
                  
                  "Iterations", CMDINTTYPE|CMDMSG, &iterations, "<count> : training/inference iterations (default 10)",
                  "it", CMDINTTYPE|CMDMSG, &iterations, "<count> : training/inference iterations (default 10)",
                  
                  "Threads", CMDINTTYPE|CMDMSG, &threads, "<count>: number of threads (default 2)",
                  "th", CMDINTTYPE|CMDMSG, &threads, "<count>: number of threads (default 2)",
                  
                  "ForceModel", CMDBOOLTYPE|CMDMSG, &forcemodel, "<bool>: force to use existing model for training",
                  "fm", CMDBOOLTYPE|CMDMSG, &forcemodel, "<bool>: force to use existing model for training",
                  
                  "MemoryMap", CMDBOOLTYPE|CMDMSG, &memorymap, "<bool>: use memory mapping (default true)",
                  "mm", CMDBOOLTYPE|CMDMSG, &memorymap, "<bool>: use memory mapping (default true)",
                  
                  "Dictionary", CMDSTRINGTYPE|CMDMSG, &dictfile, "<fname> : specify a training dictionary (optional)",
                  "d", CMDSTRINGTYPE|CMDMSG, &dictfile, "<fname> : specify training a dictionary (optional)",
                  
                  "TmpDir", CMDSTRINGTYPE|CMDMSG, &tmpdir, "<folder>: tmp directory for memory map (default /tmp)",
                  "tmp", CMDSTRINGTYPE|CMDMSG, &tmpdir, "<folder>: tmp directory for memory map (default /tmp )",

                  
                  "Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
                  "h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
                  
                  (char *)NULL
                  );
    
    if (argc == 1){
        usage();
        exit_error(IRSTLM_NO_ERROR);
    }
    
    GetParams(&argc, &argv, (char*) NULL);
    
    if (help){
        usage();
        exit_error(IRSTLM_NO_ERROR);
    }
    
    
    if (trainfile && ( !topics || !modelfile )) {
        usage();
        exit_error(IRSTLM_ERROR_DATA,"Missing training parameters");
    }
    
    if (testfile && (!modelfile || !(topicfeaturefile || wordfeaturefile))) {
        usage();
        exit_error(IRSTLM_ERROR_DATA,"Missing inference parameters");
    }
    
    dictionary *dict=NULL;
    
    //Training phase
    //test if model is readable
    bool testmodel=false;
    FILE* f;if ((f=fopen(modelfile,"r"))!=NULL){fclose(f);testmodel=true;}

    if (trainfile){
        if (testmodel){
            if (!forcemodel)
                //training with pretrained model: no need of dictionary
                exit_error(IRSTLM_ERROR_DATA,"Use -ForceModel=y option to use and update an existing model.");
        }
        else{//training with empty model and no dictionary: dictionary must be first extracted
            if (!dictfile){
                
                //    exit_error(IRSTLM_ERROR_DATA,"Missing dictionary. Provide a dictionary with option -d.");
                
                cerr << "Extracting dictionary from training data (word with freq>=" << prunethreshold << ")\n";
                dict=new dictionary(NULL,10000);
                dict->generate(trainfile,true);
                
                dictionary *sortd=new dictionary(dict,true,prunethreshold);
                sortd->sort();
                delete dict;
                dict=sortd;
                
            }
            else
                dict=new dictionary(dictfile,10000);
            dict->encode(dict->OOV());
        }
        
        plsa tc(dict,topics,tmpdir,threads,memorymap);
        tc.train(trainfile,modelfile,iterations,0.5,specialtopic);
        if (dict!=NULL) delete dict;
    }
    
    //Training phase
    //test if model is readable: notice test could be executed after training
    
    testmodel=false;
    if ((f=fopen(modelfile,"r"))!=NULL){fclose(f);testmodel=true;}
    
    if (testfile){
        if (!testmodel)
            exit_error(IRSTLM_ERROR_DATA,"Cannot read model file to run test inference.");
        if (dictfile) cerr << "Will rely on model dictionary.";

        dict=NULL;
        plsa tc(dict,topics,tmpdir,threads,memorymap);
        tc.inference(testfile,modelfile,iterations,topicfeaturefile,wordfeaturefile);
        if (dict!=NULL) delete dict;
    }
    
    
    //save/convert model in text format
    
    if (txtfile){
        if (!testmodel)
            exit_error(IRSTLM_ERROR_DATA,"Cannot open model to be printed in readable format.");

        dict=NULL;
        plsa tc(dict,topics,tmpdir,threads,memorymap);
        tc.initW(modelfile,1,0);
        tc.saveWtxt(txtfile,topwords);
        tc.freeW();
    }
    
    exit_error(IRSTLM_NO_ERROR);
}



