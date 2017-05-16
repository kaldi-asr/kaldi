// $Id: dictionary.cpp 3640 2010-10-08 14:58:17Z bertoldi $

/******************************************************************************
 IrstLM: IRST Language Model Toolkit
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

#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include "mempool.h"
#include "htable.h"
#include "index.h"
#include "util.h"
#include "dictionary.h"
#include "mfstream.h"

using namespace std;

dictionary::dictionary(char *filename,int size, float lf)
{
	if (lf<=0.0) lf=DICTIONARY_LOAD_FACTOR;
	load_factor=lf;
	
	htb = new HASHTABLE_t((size_t) (size/load_factor));
	tb  = new dict_entry[size];
	st  = new strstack(size * 10);
	
	for (int i=0; i<size; i++) tb[i].freq=0;
	
	oov_code = -1;
	
	n  = 0;
	N  =  0;
	dubv = 0;
	lim = size;
	ifl=0;  //increment flag
	
	if (filename==NULL) return;
	
	mfstream inp(filename,ios::in);
	
	if (!inp) {
		std::stringstream ss_msg;
		ss_msg << "cannot open " << filename << "\n";
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
	}
	
	char buffer[100];
	
	inp >> setw(100) >> buffer;
	
	inp.close();
	
	if ((strncmp(buffer,"dict",4)==0) ||
			(strncmp(buffer,"DICT",4)==0))
		load(filename);
	else
		generate(filename);
	
	cerr << "loaded \n";
	
}


int dictionary::getword(fstream& inp , char* buffer) const
{
	while(inp >> setw(MAX_WORD) >> buffer) {
		
		//warn if the word is very long
		if (strlen(buffer)==(MAX_WORD-1)) {
			cerr << "getword: a very long word was read ("
			<< buffer << ")\n";
		}
		
		//skip words of length zero chars: why should this happen?
		if (strlen(buffer)==0) {
			cerr << "zero length word!\n";
			continue;
		}
		
		return 1;
	}
	
	return 0;
}


void dictionary::generate(char *filename,bool header)
{
	
	char buffer[MAX_WORD];
	int counter=0;
	
	mfstream inp(filename,ios::in);
	
	if (!inp) {
		std::stringstream ss_msg;
		ss_msg << "cannot open " << filename << "\n";
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
	}
	
	cerr << "dict:";
	
	ifl=1;

        //skip header
	if (header) inp.getline(buffer,MAX_WORD);
	
	while (getword(inp,buffer)) {
		
		incfreq(encode(buffer),1);
		
		if (!(++counter % 1000000)) cerr << ".";
	}
	
	ifl=0;
	
	cerr << "\n";
	
	inp.close();
	
}

void dictionary::augment(dictionary *d)
{
	incflag(1);
	for (int i=0; i<d->n; i++)
		encode(d->decode(i));
	incflag(0);
	encode(OOV());
}


// print_curve: show statistics on dictionary growth
void dictionary::print_curve_growth(int curvesize) const
{
        int* curve = new int[curvesize];
        for (int i=0; i<curvesize; i++) curve[i]=0;

        // filling the curve
        for (int i=0; i<n; i++) {
                if(tb[i].freq > curvesize-1)
                        curve[curvesize-1]++;
                else
                        curve[tb[i].freq-1]++;
        }

        //cumulating results
        for (int i=curvesize-2; i>=0; i--) {
                curve[i] = curve[i] + curve[i+1];
        }

        cout.setf(ios::fixed);
        cout << "Dict size: " << n << "\n";
        cout << "**************** DICTIONARY GROWTH CURVE ****************\n";
        cout << "Freq\tEntries\tPercent";
        cout << "\n";

        for (int i=0; i<curvesize; i++) {
                cout << ">" << i << "\t" << curve[i] << "\t" << setprecision(2) << (float)curve[i]/n * 100.0 << "%";
                cout << "\n";
        }
        cout << "*********************************************************\n";
        delete []curve;
}

// print_curve_oov: show OOV amount and OOV rates computed on test corpus
void dictionary::print_curve_oov(int curvesize, const char *filename, int listflag)
{
    int *OOVchart=new int[curvesize];
	int NwTest;

        test(OOVchart, &NwTest, curvesize, filename, listflag);

	cout.setf(ios::fixed);
        cout << "Dict size: " << n << "\n";
        cout << "Words of test: " << NwTest << "\n";
        cout << "**************** OOV RATE STATISTICS ****************\n";
        cout << "Freq\tOOV_Entries\tOOV_Rate";
        cout << "\n";

        for (int i=0; i<curvesize; i++) {

                // display OOV iamount and OOV rates on test
                cout << "<" << i+1 << "\t" << OOVchart[i] << "\t" << setprecision(2) << (float)OOVchart[i]/NwTest * 100.0 << "%";
                cout << "\n";
        }
        cout << "*********************************************************\n";
        delete []OOVchart;
}

//
//      test : compute OOV rates on test corpus using dictionaries of different sizes
//
void dictionary::test(int* OOVchart, int* NwTest, int curvesize, const char *filename, int listflag)
{
        MY_ASSERT(OOVchart!=NULL);

        int m_NwTest=0;
        for (int j=0; j<curvesize; j++) OOVchart[j]=0;
        char buffer[MAX_WORD];

        const char* bos = BoS();

        mfstream inp(filename,ios::in);

        if (!inp) {
                std::stringstream ss_msg;
                ss_msg << "cannot open " << filename << "\n";
                exit_error(IRSTLM_ERROR_IO, ss_msg.str());
        }
        cerr << "test:";

        int k = 0;
        while (getword(inp,buffer)) {

                // skip 'beginning of sentence' symbol
                if (strcmp(buffer,bos)==0)
                        continue;

                int freq = 0;
                int wCode = getcode(buffer);
                if(wCode!=-1) freq = tb[wCode].freq;

                if(freq==0) {
                        OOVchart[0]++;
                        if(listflag) {
                                cerr << "<OOV>" << buffer << "</OOV>\n";
                        }
                } else {
                        if(freq < curvesize) OOVchart[freq]++;
                }
                m_NwTest++;
                if (!(++k % 1000000)) cerr << ".";
        }
        cerr << "\n";
        inp.close();

        // cumulating results
        for (int i=1; i<curvesize; i++){
                OOVchart[i] = OOVchart[i] + OOVchart[i-1];
        }
        *NwTest=m_NwTest;
}

void dictionary::load(char* filename)
{
	char header[100];
	char* addr;
	char buffer[MAX_WORD];
	int freqflag=0;
	
	mfstream inp(filename,ios::in);
	
	if (!inp) {
		std::stringstream ss_msg;
		ss_msg << "cannot open " << filename << "\n";
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
	}
	
	cerr << "dict:";
	
	inp.getline(header,100);
	if (strncmp(header,"DICT",4)==0)
		freqflag=1;
	else if (strncmp(header,"dict",4)!=0) {
		std::stringstream ss_msg;
		ss_msg << "dictionary file " << filename << " has a wrong header";
		exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
	}
	
	
	while (getword(inp,buffer)) {
		
		tb[n].word=st->push(buffer);
		tb[n].code=n;
		
		if (freqflag)
			inp >> tb[n].freq;
		else
			tb[n].freq=0;
		
		//always insert without checking whether the word is already in
		if ((addr=htb->insert((char*)&tb[n].word))) {
			if (addr!=(char *)&tb[n].word) {
				cerr << "dictionary::loadtxt wrong entry was found ("
				<<  buffer << ") in position " << n << "\n";
				//      exit(1);
				continue;  // continue loading dictionary
			}
		}
		
		N+=tb[n].freq;
		
		if (strcmp(buffer,OOV())==0) oov_code=n;
		
		if (++n==lim) grow();
		
	}
	
	inp.close();
}


void dictionary::load(std::istream& inp)
{
	
	char buffer[MAX_WORD];
	char *addr;
	int size;
	
	inp >> size;
	
	for (int i=0; i<size; i++) {
		
		inp >> setw(MAX_WORD) >> buffer;
		
		tb[n].word=st->push(buffer);
		tb[n].code=n;
		inp >> tb[n].freq;
		N+=tb[n].freq;
		
		//always insert without checking whether the word is already in
		if ((addr=htb->insert((char  *)&tb[n].word))) {
			if (addr!=(char *)&tb[n].word) {
				std::stringstream ss_msg;
				ss_msg << "dictionary::loadtxt wrong entry was found (" <<  buffer << ") in position " << n;
				exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
			}
		}
		
		if (strcmp(tb[n].word,OOV())==0)
			oov_code=n;
		
		if (++n==lim) grow();
	}
	inp.getline(buffer,MAX_WORD-1);
}


void dictionary::save(std::ostream& out)
{
	out << n << "\n";
	for (int i=0; i<n; i++)
		out << tb[i].word << " " << tb[i].freq << "\n";
}

int cmpdictentry(const void *a,const void *b)
{
	dict_entry *ae=(dict_entry *)a;
	dict_entry *be=(dict_entry *)b;
	
	if (be->freq-ae->freq)
		return be->freq-ae->freq;
	else
		return strcmp(ae->word,be->word);
	
}


dictionary::dictionary(dictionary* d,bool prune, int prunethresh)
{
	MY_ASSERT(d!=NULL);
	//transfer values
	n=0;        //total entries
	N=0;        //total frequency
	
	load_factor=d->load_factor;        //load factor
	lim=d->lim;    //limit of entries
	oov_code=-1;   //code od oov must be re-defined	
	ifl=0;         //increment flag=0;
	dubv=d->dubv;  //dictionary upperbound transferred
	
	//creates a sorted copy of the table
	tb  = new dict_entry[lim];
	htb = new HASHTABLE_t((size_t) (lim/load_factor));
	st  = new strstack(lim * 10);
	
	//copy in the entries with frequency > threshold
	n=0;
	for (int i=0; i<d->n; i++)
		if (!prune || d->tb[i].freq>prunethresh){
			tb[n].code=n;
			tb[n].freq=d->tb[i].freq;
			tb[n].word=st->push(d->tb[i].word);
			htb->insert((char*)&tb[n].word);
			
			if (d->oov_code==i) oov_code=n; //reassign oov_code
			
			N+=tb[n].freq;
			n++;
		}			
};

void dictionary::sort()
{
	if (htb != NULL )  delete htb;
	
	htb = new HASHTABLE_t((int) (lim/load_factor));
	//sort all entries according to frequency
	cerr << "sorting dictionary ...";
	qsort(tb,n,sizeof(dict_entry),cmpdictentry);
	cerr << "done\n";
	
	for (int i=0; i<n; i++) {
		//eventually re-assign oov code
		if (oov_code==tb[i].code) oov_code=i;
		tb[i].code=i;
		//always insert without checking whether the word is already in
		htb->insert((char*)&tb[i].word);
	};
	
}

dictionary::~dictionary()
{
	delete htb;
	delete st;
	delete [] tb;
}

void dictionary::stat() const
{
	cout << "dictionary class statistics\n";
	cout << "size " << n
	<< " used memory "
	<< (lim * sizeof(int) +
			htb->used() +
			st->used())/1024 << " Kb\n";
}

void dictionary::grow()
{
	delete htb;
	
	cerr << "+\b";
	
	int newlim=(int) (lim*GROWTH_STEP);
	dict_entry *tb2=new dict_entry[newlim];
	
	memcpy(tb2,tb,sizeof(dict_entry) * lim );
	
	delete [] tb;
	tb=tb2;
	
	htb=new HASHTABLE_t((size_t) ((newlim)/load_factor));
	for (int i=0; i<lim; i++) {
		//always insert without checking whether the word is already in
		htb->insert((char*)&tb[i].word);
	}
	
	for (int i=lim; i<newlim; i++) tb[i].freq=0;
	
	lim=newlim;
}

void dictionary::save(char *filename,int freqflag)
{
	
	std::ofstream out(filename,ios::out);
	
	if (!out) {
		cerr << "cannot open " << filename << "\n";
	}
	
	// header
	if (freqflag)
		out << "DICTIONARY 0 " << n << "\n";
	else
		out << "dictionary 0 " << n << "\n";
	
	for (int i=0; i<n; i++)
		if (tb[i].freq) { //do not print pruned words!
			out << tb[i].word;
			if (freqflag)
				out << " " << tb[i].freq;
			out << "\n";
		}
	
	out.close();
}


int dictionary::getcode(const char *w)
{
	dict_entry* ptr=(dict_entry *)htb->find((char *)&w);
	if (ptr==NULL) return -1;
	return ptr->code;
}

int dictionary::encode(const char *w)
{
	//case of strange characters
	if (strlen(w)==0) {
		cerr << "0";
		w=OOV();
	}
	
	
	dict_entry* ptr;
	
	if ((ptr=(dict_entry *)htb->find((char *)&w))!=NULL)
		return ptr->code;
	else {
		if (!ifl) { //do not extend dictionary
			if (oov_code==-1) { //did not use OOV yet
				cerr << "starting to use OOV words [" << w << "]\n";
				tb[n].word=st->push(OOV());
				htb->insert((char  *)&tb[n].word);
				tb[n].code=n;
				tb[n].freq=0;
				oov_code=n;
				if (++n==lim) grow();
			}
			
			return encode(OOV());
		} else { //extend dictionary
			tb[n].word=st->push((char *)w);
			htb->insert((char*)&tb[n].word);
			tb[n].code=n;
			tb[n].freq=0;
			if (++n==lim) grow();
			return n-1;
		}
	}
}


const char *dictionary::decode(int c) const
{
	if (c>=0 && c < n)
		return tb[c].word;
	else {
		cerr << "decode: code out of boundary\n";
		return OOV();
	}
}


dictionary_iter::dictionary_iter(dictionary *dict) : m_dict(dict)
{
	m_dict->scan(HT_INIT);
}

dict_entry* dictionary_iter::next()
{
	return  (dict_entry*) m_dict->scan(HT_CONT);
}

/*
 main(int argc,char **argv){
 dictionary d(argv[1],40000);
 d.stat();
 cout << "ROMA" << d.decode(0) << "\n";
 cout << "ROMA:" << d.encode("ROMA") << "\n";
 d.save(argv[2]);
 }
 */
