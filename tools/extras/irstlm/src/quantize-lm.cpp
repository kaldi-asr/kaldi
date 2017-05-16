// $Id: quantize-lm.cpp 302 2009-08-25 13:04:13Z nicolabertoldi $

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
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include "cmd.h"
#include "math.h"
#include "util.h"
#include "mfstream.h"


using namespace std;

//----------------------------------------------------------------------
//  Special type and global variable for the BIN CLUSTERING algorithm
//
//
//----------------------------------------------------------------------


typedef struct {
  float pt;
  unsigned int idx;
  unsigned short code;
} DataItem;


int cmpFloatEntry(const void* a,const void* b)
{
  if (*(float *)a > *(float*)b)
    return 1;
  else if (*(float *)a < *(float *)b)
    return -1;
  else
    return 0;
}

//----------------------------------------------------------------------
//  Global entry points
//----------------------------------------------------------------------

int ComputeCluster(int nc, double* cl,unsigned int N,DataItem* Pts);

//----------------------------------------------------------------------
//  Global parameters (some are set in getArgs())
//----------------------------------------------------------------------

int       k      = 256;   // number of centers
const int MAXLEV = 11;    //maximum n-gram size

//----------------------------------------------------------------------
//  Main program
//----------------------------------------------------------------------

void print_help(int TypeFlag=0){
  std::cerr << std::endl << "quantize-lm - quantizes probabilities and back-off weights" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       quantize-lm <input-file.lm> [<output-file.qlm> [<tmpfile>]]" << std::endl;
  std::cerr << std::endl << "DESCRIPTION:" << std::endl;
	std::cerr << "       quantize-lm reads a standard LM file in ARPA format and produces" << std::endl;
	std::cerr << "       a version of it with quantized probabilities and back-off weights"<< std::endl;
	std::cerr << "       that the IRST LM toolkit can compile. Accepts LMs with .gz suffix." << std::endl;
	std::cerr << "       You can specify the output file to be created and also the pathname" << std::endl;
	std::cerr << "       of a temporary file used by the program. As default, the temporary "  << std::endl;
	std::cerr << "       file is created in the /tmp directory." << std::endl;
	std::cerr << "       Output file can be written to standard output by using the special name -."  << std::endl;
  std::cerr << std::endl << "OPTIONS:" << std::endl;
	
	FullPrintParams(TypeFlag, 0, 1, stderr);
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

int main(int argc, char **argv)
{

  std::vector<std::string> files;
	
	bool help=false;
	
	DeclareParams((char*)
								"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
								(char *)NULL
								);
	
	if (argc == 1){
		usage();
	}
	
	int first_file=1;
	for (int i=1; i < argc; i++) {
		if (strcmp(argv[i],"-") == 0){ //handles /dev/stdin or /dev/stdout
			if (first_file == 1){
				files.push_back("/dev/stdin");
			}else if (first_file == 2){
				files.push_back("/dev/stdout");
			}else{
				usage("Warning: You can use the value for the input and/or output file only");
			}
			first_file++;
		}else if(argv[i][0] != '-'){
			files.push_back(argv[i]);
			first_file++;
		}
	}
			 
	GetParams(&argc, &argv, (char*) NULL);
	
	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
  if (files.size() > 3) {
		exit_error(IRSTLM_ERROR_DATA,"Too many arguments");
  }

  if (files.size() < 1) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Please specify a LM file to read from");
  }

  std::string infile = files[0];
  std::string outfile="";
  std::string tmpfile="";

  if (files.size() == 1) {

    outfile=infile;

    //remove path information
    std::string::size_type p = outfile.rfind('/');
    if (p != std::string::npos && ((p+1) < outfile.size()))
      outfile.erase(0,p+1);

    //eventually strip .gz
    if (outfile.compare(outfile.size()-3,3,".gz")==0)
      outfile.erase(outfile.size()-3,3);

    outfile+=".qlm";
  } else
    outfile = files[1];


  if (files.size()==3) {
    //create temporary file
    tmpfile = files[2];
    mfstream dummy(tmpfile.c_str(),ios::out);
    dummy.close();
  } else {
    //create temporary internal file in /tmp
    mfstream dummy;
    createtempfile(dummy,tmpfile,ios::out);
    dummy.close();
  }

  std::cerr << "Reading " << infile << "..." << std::endl;

  inputfilestream inp(infile.c_str());
	if (!inp.good()) {
		std::stringstream ss_msg;
		ss_msg << "Failed to open " << infile;;
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
  }
	
  std::ofstream* out;
  if (outfile == "-")
    out = (ofstream *)&std::cout;
  else {
    out=new std::ofstream;
    out->open(outfile.c_str());
  }
	if (!out->good()) {
		std::stringstream ss_msg;
		ss_msg << "Failed to open " << outfile;
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
  }

  std::cerr << "Writing " << outfile << "..." << std::endl;

  //prepare temporary file to save n-gram blocks for multiple reads
  //this avoids using seeks which do not work with inputfilestream
  //it's odd but i need a bidirectional filestream!
  std::cerr << "Using temporary file " << tmpfile << std::endl;
  fstream filebuff(tmpfile.c_str(),ios::out|ios::in|ios::binary);

  unsigned int nPts = 0;  // actual number of points

  // *** Read ARPA FILE **

  unsigned int numNgrams[MAXLEV + 1]; /* # n-grams for each order */
  int Order=0,MaxOrder=0;
  int n=0;

  float logprob,logbow;

  DataItem* dataPts;

  double* centersP=NULL;
  double* centersB=NULL;

  //maps from point index to code
  unsigned short* mapP=NULL;
  unsigned short* mapB=NULL;

  int centers[MAXLEV + 1];
  streampos iposition;

  for (int i=1; i<=MAXLEV; i++) numNgrams[i]=0;
  for (int i=1; i<=MAXLEV; i++) centers[i]=k;

  /* all levels 256 centroids; in case read them as parameters */

  char line[MAX_LINE];

  while (inp.getline(line,MAX_LINE)) {

    bool backslash = (line[0] == '\\');

    if (sscanf(line, "ngram %d=%d", &Order, &n) == 2) {
      numNgrams[Order] = n;
      MaxOrder=Order;
      continue;
    }

    if (!strncmp(line, "\\data\\", 6) || strlen(line)==0)
      continue;

    if (backslash && sscanf(line, "\\%d-grams", &Order) == 1) {

      // print output header:
      if (Order == 1) {
        *out << "qARPA " << MaxOrder;
        for (int i=1; i<=MaxOrder; i++)
          *out << " " << centers[i];
        *out << "\n\n\\data\\\n";

        for (int i=1; i<=MaxOrder; i++)
          *out << "ngram " << i << "= " << numNgrams[i] << "\n";
      }

      *out << "\n";
      *out << line << "\n";
      cerr << "-- Start processing of " << Order << "-grams\n";
      MY_ASSERT(Order <= MAXLEV);

      unsigned int N=numNgrams[Order];

      const char* words[MAXLEV+3];
      dataPts=new DataItem[N]; // allocate data

      //reset tempout file to start writing
      filebuff.seekg((streampos)0);

      for (nPts=0; nPts<N; nPts++) {
        inp.getline(line,MAX_LINE);
        filebuff << line << std::endl;
        if (!filebuff.good()) {
          removefile(tmpfile.c_str());
					std::stringstream ss_msg;
					ss_msg << "Cannot write in temporary file " << tmpfile  << std::endl
					<< " Probably there is not enough space in this filesystem " << std::endl
					<< " Eventually rerun quantize-lm by specifyng the pathname" << std::endl
					<< " of the temporary file to be used. ";
					exit_error(IRSTLM_ERROR_IO, ss_msg.str());
        }
        int howmany = parseWords(line, words, Order + 3);
        MY_ASSERT(howmany == Order+2 || howmany == Order+1);
        sscanf(words[0],"%f",&logprob);
        dataPts[nPts].pt=logprob; //exp(logprob * logten);
        dataPts[nPts].idx=nPts;
      }

      cerr << "quantizing " << N << " probabilities\n";

      centersP=new double[centers[Order]];
      mapP=new unsigned short[N];

      ComputeCluster(centers[Order],centersP,N,dataPts);


      for (unsigned int p=0; p<N; p++) {
        mapP[dataPts[p].idx]=dataPts[p].code;
      }

      if (Order<MaxOrder) {
        //second pass to read back-off weights
        //read from temporary file
        filebuff.seekg((streampos)0);

        for (nPts=0; nPts<N; nPts++) {

          filebuff.getline(line,MAX_LINE);
          int howmany = parseWords(line, words, Order + 3);
          if (howmany==Order+2) //backoff is written
            sscanf(words[Order+1],"%f",&logbow);
          else
            logbow=0; // backoff is implicit

          dataPts[nPts].pt=logbow;
          dataPts[nPts].idx=nPts;
        }

        centersB=new double[centers[Order]];
        mapB=new unsigned short[N];

        cerr << "quantizing " << N << " backoff weights\n";
        ComputeCluster(centers[Order],centersB,N,dataPts);

        for (unsigned int p=0; p<N; p++) {
          mapB[dataPts[p].idx]=dataPts[p].code;
        }

      }


      *out << centers[Order] << "\n";
      for (int c=0; c<centers[Order]; c++) {
        *out << centersP[c];
        if (Order<MaxOrder) *out << " " << centersB[c];
        *out << "\n";
      }

      filebuff.seekg(0);

      for (nPts=0; nPts<numNgrams[Order]; nPts++) {

        filebuff.getline(line,MAX_LINE);

        parseWords(line, words, Order + 3);

        *out << mapP[nPts];

        for (int i=1; i<=Order; i++) *out << "\t" << words[i];

        if (Order < MaxOrder) *out << "\t" << mapB[nPts];

        *out << "\n";

      }

      if (mapP) {
        delete [] mapP;
        mapP=NULL;
      }
      if (mapB) {
        delete [] mapB;
        mapB=NULL;
      }

      if (centersP) {
        delete [] centersP;
        centersP=NULL;
      }
      if (centersB) {
        delete [] centersB;
        centersB=NULL;
      }

      delete [] dataPts;

      continue;


    }

  }

  *out << "\\end\\\n";
  cerr << "---- done\n";

  out->flush();

  out->close();
  inp.close();

  removefile(tmpfile.c_str());
}

// Compute Clusters

int ComputeCluster(int centers,double* ctrs,unsigned int N,DataItem* bintable)
{


  //cerr << "\nExecuting Clutering Algorithm:  k=" << centers<< "\n";
  double log10=log(10.0);

  for (unsigned int i=0; i<N; i++) bintable[i].code=0;

  //cout << "start sort \n";
  qsort(bintable,N,sizeof(DataItem),cmpFloatEntry);

  unsigned int different=1;

  for (unsigned int i=1; i<N; i++)
    if (bintable[i].pt!=bintable[i-1].pt)
      different++;

  unsigned int interval=different/centers;
  if (interval==0) interval++;

  unsigned int* population=new unsigned int[centers];
  unsigned int* species=new unsigned int[centers];

  //cerr << " Different entries=" << different
  //     << " Total Entries=" << N << " Bin Size=" << interval << "\n";

  for (int i=0; i<centers; i++) {
    population[i]=species[i]=0;
    ctrs[i]=0;
  }

  // initial values: this should catch up very low values: -99
  bintable[0].code=0;
  population[0]=1;
  species[0]=1;

  int currcode=0;
  different=1;

  for (unsigned int i=1; i<N; i++) {

    if ((bintable[i].pt!=bintable[i-1].pt)) {
      different++;
      if ((different % interval) == 0)
        if ((currcode+1) < centers
            &&
            population[currcode]>0) {
          currcode++;
        }
    }

    if (bintable[i].pt == bintable[i-1].pt)
      bintable[i].code=bintable[i-1].code;
    else {
      bintable[i].code=currcode;
      species[currcode]++;
    }

    population[bintable[i].code]++;

    MY_ASSERT(bintable[i].code < centers);

    ctrs[bintable[i].code]=ctrs[bintable[i].code]+exp(bintable[i].pt * log10);

  }

  for (int i=0; i<centers; i++) {
    if (population[i]>0)
      ctrs[i]=log(ctrs[i]/population[i])/log10;
    else
      ctrs[i]=-99;

    if (ctrs[i]<-99) {
      cerr << "Warning: adjusting center with too small prob " << ctrs[i] << "\n";
      ctrs[i]=-99;
    }

    cerr << i << " ctr " << ctrs[i] << " population " << population[i] << " species " << species[i] <<"\n";
  }

  cout.flush();

  delete [] population;
  delete [] species;


  return 1;

}

//----------------------------------------------------------------------
//  Reading/Printing utilities
//      readPt - read a point from input stream into data storage
//              at position i.  Returns false on error or EOF.
//      printPt - prints a points to output file
//----------------------------------------------------------------------

