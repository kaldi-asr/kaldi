// $Id: dictionary.h 3679 2010-10-13 09:10:01Z bertoldi $

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

#ifndef MF_DICTIONARY_H
#define MF_DICTIONARY_H

#include "mfstream.h"
#include "htable.h"
#include <cstring>
#include <iostream>


using namespace std;

#define MAX_WORD 1000
#define DICTIONARY_LOAD_FACTOR  2.0


#ifndef GROWTH_STEP
#define GROWTH_STEP 1.5
#endif

#ifndef DICT_INITSIZE
#define DICT_INITSIZE 100000
#endif

//Begin of sentence symbol
#ifndef BOS_
#define BOS_ "<s>"
#endif


//End of sentence symbol
#ifndef EOS_
#define EOS_ "</s>"
#endif

//End of document symbol
#ifndef BOD_
#define BOD_ "<d>"
#endif

//End of document symbol
#ifndef EOD_
#define EOD_ "</d>"
#endif


//Out-Of-Vocabulary symbol
#ifndef OOV_
#define OOV_ "<unk>"
#endif

typedef struct {
	const char *word;
	int  code;
	long long freq;
} dict_entry;

typedef htable<char*> HASHTABLE_t;

class strstack;

class dictionary
{
	strstack   *st;  //!< stack of strings
	dict_entry *tb;  //!< entry table
	HASHTABLE_t  *htb;  //!< hash table
	int          n;  //!< number of entries
	long long    N;  //!< total frequency
	int        lim;  //!< limit of entries
	int   oov_code;  //!< code assigned to oov words
	char       ifl;  //!< increment flag
	int        dubv; //!< dictionary size upper bound
	float        load_factor; //!< dictionary loading factor
	
	void test(int* OOVchart, int* NwTest, int curvesize, const char *filename, int listflag=0);	// prepare into testOOV the OOV statistics computed on test set
	
public:
	
	friend class dictionary_iter;
	
	dictionary* oovlex; //<! additional dictionary
	
	inline int dub() const {
		return dubv;
	}
	
	inline int dub(int value) {
		return dubv=value;
	}
	
	inline static const char *OOV() {
		return (char*) OOV_;
	}
	
	inline static const char *BoS() {
		return (char*) BOS_;
	}
	
	inline static const char *EoS() {
		return (char*) EOS_;
	}
	
	inline static const char *BoD() {
		return (char*) BOD_;
	}
	
	inline static const char *EoD() {
		return (char*) EOD_;
	}
	
	inline int oovcode(int v=-1) {
		return oov_code=(v>=0?v:oov_code);
	}
	
	inline int incflag() const {
		return ifl;
	}
	
	inline int incflag(int v) {
		return ifl=v;
	}
	
	int getword(fstream& inp , char* buffer) const;
	
	int isprintable(char* w) const {
		char buffer[MAX_WORD];
		sprintf(buffer,"%s",w);
		return strcmp(w,buffer)==0;
	}
	
	inline void genoovcode() {
		int c=encode(OOV());
		std::cerr << "OOV code is "<< c << std::endl;
		cerr << "OOV code is "<< c << std::endl;
		oovcode(c);
	}
	
	inline void genBoScode() {
		int c=encode(BoS());
		std::cerr << "BoS code is "<< c << std::endl;
	}
	
	inline void genEoScode() {
		int c=encode(EoS());
		std::cerr << "EoS code is "<< c << std::endl;
	}
	
	inline long long setoovrate(double oovrate) {
		encode(OOV()); //be sure OOV code exists
		long long oovfreq=(long long)(oovrate * totfreq());
		std::cerr << "setting OOV rate to: " << oovrate << " -- freq= " << oovfreq << std::endl;
		return freq(oovcode(),oovfreq);
	}
	
	inline long long incfreq(int code,long long value) {
		N+=value;
		return tb[code].freq+=value;
	}
	
	inline long long multfreq(int code,double value) {
		N+=(long long)(value * tb[code].freq)-tb[code].freq;
		return tb[code].freq=(long long)(value * tb[code].freq);
	}
	
	inline long long freq(int code,long long value=-1) {
		if (value>=0) {
			N+=value-tb[code].freq;
			tb[code].freq=value;
		}
		return tb[code].freq;
	}
	
	inline long long totfreq() const {
		return N;
	}
	
	inline float set_load_factor(float value) {
		return load_factor=value;
	}
	
	void grow();
	void sort();
	
	dictionary(char *filename,int size=DICT_INITSIZE,float lf=DICTIONARY_LOAD_FACTOR);
	dictionary(dictionary* d, bool prune=false,int prunethresh=0); //make a copy and eventually filter out unfrequent words
	
	~dictionary();
	void generate(char *filename,bool header=false);
	void load(char *filename);
	void save(char *filename, int freqflag=0);
	void load(std::istream& fd);
	void save(std::ostream& fd);
	
	void augment(dictionary *d);
	
	int size() const {
		return n;
	}
	int getcode(const char *w);
	int encode(const char *w);
	
	const char *decode(int c) const;
	void stat() const;
	
	void print_curve_growth(int curvesize) const;
	void print_curve_oov(int curvesize, const char *filename, int listflag=0);
	
	void cleanfreq() {
		for (int i=0; i<n; ++i){ tb[i].freq=0; };
		N=0;
	}
	
	inline dict_entry* scan(HT_ACTION action) {
		return  (dict_entry*) htb->scan(action);
	}
};

class dictionary_iter
{
public:
	dictionary_iter(dictionary *dict);
	dict_entry* next();
private:
	dictionary* m_dict;
};
#endif

