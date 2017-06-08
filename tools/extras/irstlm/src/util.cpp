// $Id: util.cpp 363 2010-02-22 15:02:45Z mfederico $
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

#ifdef WIN32
#include <windows.h>
#include <string.h>
#include <io.h>
#else
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/mman.h>
#endif
#include "gzfilebuf.h"
#include "timer.h"
#include "util.h"
#include "n_gram.h"
#include "mfstream.h"

using namespace std;

string gettempfolder()
{
#ifdef _WIN32
	char *tmpPath = getenv("TMP");
	string str(tmpPath);
	if (str.substr(str.size() - 1, 1) != "\\")
		str += "\\";
	return str;
#else
	char *tmpPath = getenv("TMP");
	if (!tmpPath || !*tmpPath)
		return "/tmp/";
	string str(tmpPath);
	if (str.substr(str.size() - 1, 1) != "/")
		str += "/";
	return str;
#endif
}

string createtempName()
{       
	string tmpfolder = gettempfolder();
#ifdef _WIN32
	char buffer[BUFSIZ];
	//To check whether the following function open the stream as well
	//In this case it is mandatory to close it immediately
	::GetTempFileNameA(tmpfolder.c_str(), "", 0, buffer);
#else
	char buffer[tmpfolder.size() + 16];
	strcpy(buffer, tmpfolder.c_str());
	strcat(buffer, "dskbuff--XXXXXX");
	int fd=mkstemp(buffer);
	close(fd);
#endif
	return (string) buffer;
}

void createtempfile(mfstream  &fileStream, string &filePath, std::ios_base::openmode flags)
{       
	filePath = createtempName();
	fileStream.open(filePath.c_str(), flags);
}

void removefile(const std::string &filePath)
{
#ifdef _WIN32
	::DeleteFileA(filePath.c_str());
#else
	if (remove(filePath.c_str()) != 0)
	{
		perror("Error deleting file" );
		exit_error(IRSTLM_ERROR_IO);
	}
#endif
}

/* MemoryMap Management
 Code kindly provided by Fabio Brugnara, ITC-irst Trento.
 How to use it:
 - call MMap with offset and required size (psgz):
 pg->b = MMap(fd, rdwr,offset,pgsz,&g);
 - correct returned pointer with the alignment gap and save the gap:
 pg->b += pg->gap = g;
 - when releasing mapped memory, subtract the gap from the pointer and add
 the gap to the requested dimension
 Munmap(pg->b-pg->gap, pgsz+pg->gap, 0);
 */


void *MMap(int	fd, int	access, off_t	offset, size_t	len, off_t	*gap)
{
	void	*p=NULL;
	
#ifdef _WIN32
	/*
	 int g=0;
	 // code for windows must be checked
	 HANDLE	fh,
	 mh;
	 
	 fh = (HANDLE)_get_osfhandle(fd);
	 if(offset) {
	 // bisogna accertarsi che l'offset abbia la granularita`
	 //corretta, MAI PROVATA!
	 SYSTEM_INFO	si;
	 
	 GetSystemInfo(&si);
	 g = *gap = offset % si.dwPageSize;
	 } else if(gap) {
	 *gap=0;
	 }
	 if(!(mh=CreateFileMapping(fh, NULL, PAGE_READWRITE, 0, len+g, NULL))) {
	 return 0;
	 }
	 p = (char*)MapViewOfFile(mh, FILE_MAP_ALL_ACCESS, 0,
	 offset-*gap, len+*gap);
	 CloseHandle(mh);
	 */
	
#else
	int pgsz,g=0;
	if(offset) {
		pgsz = sysconf(_SC_PAGESIZE);
		g = *gap = offset%pgsz;
	} else if(gap) {
		*gap=0;
	}
	p = mmap((void*)0, len+g, access,
					 MAP_SHARED|MAP_FILE,
					 fd, offset-g);
	if((long)p==-1L)
	{
		perror("mmap failed");
		p=0;
	}
#endif
	return p;
}


int Munmap(void	*p,size_t	len,int	sync)
{
	int	r=0;
	
#ifdef _WIN32
	/*
	 //code for windows must be checked
	 if(sync) FlushViewOfFile(p, len);
	 UnmapViewOfFile(p);
	 */
#else
	cerr << "len  = " << len << endl;
	cerr << "sync = " << sync << endl;
	cerr << "running msync..." << endl;
	if(sync) msync(p, len, MS_SYNC);
	cerr << "done. Running munmap..." << endl;
	if((r=munmap((void*)p, len)))
	{
		perror("munmap() failed");
	}
	cerr << "done" << endl;
	
#endif
	return r;
}


//global variable
Timer g_timer;


void ResetUserTime()
{
	g_timer.start();
};

void PrintUserTime(const std::string &message)
{
	g_timer.check(message.c_str());
}

double GetUserTime()
{
	return g_timer.get_elapsed_time();
}


void ShowProgress(long long current, long long target){
    
    int frac=(current * 1000)/target;
    if (!(frac % 10)) fprintf(stderr,"%02d\b\b",frac/10);

}


int parseWords(char *sentence, const char **words, int max)
{
	char *word;
	int i = 0;
	
	const char *const wordSeparators = " \t\r\n";
	
	for (word = strtok(sentence, wordSeparators);
			 i < max && word != 0;
			 i++, word = strtok(0, wordSeparators)) {
		words[i] = word;
	}
	
	if (i < max) {
		words[i] = 0;
	}
	
	return i;
}


//Load a LM as a text file. LM could have been generated either with the
//IRST LM toolkit or with the SRILM Toolkit. In the latter we are not
//sure that n-grams are lexically ordered (according to the 1-grams).
//However, we make the following assumption:
//"all successors of any prefix are sorted and written in contiguous lines!"
//This method also loads files processed with the quantization
//tool: qlm

int parseline(istream& inp, int Order,ngram& ng,float& prob,float& bow)
{
	
	const char* words[1+ LMTMAXLEV + 1 + 1];
	int howmany;
	char line[MAX_LINE];
	
	inp.getline(line,MAX_LINE);
	if (strlen(line)==MAX_LINE-1) {
		std::stringstream ss_msg;
		ss_msg << "parseline: input line exceed MAXLINE (" << MAX_LINE << ") chars " << line << "\n";
		
		exit_error(IRSTLM_ERROR_DATA, ss_msg.str());
	}
	
	howmany = parseWords(line, words, Order + 3);
	
	if (!(howmany == (Order+ 1) || howmany == (Order + 2))){
		MY_ASSERT(howmany == (Order+ 1) || howmany == (Order + 2));
	}
	
	//read words
	ng.size=0;
	for (int i=1; i<=Order; i++)
		ng.pushw(strcmp(words[i],"<unk>")?words[i]:ng.dict->OOV());
	
	//read logprob/code and logbow/code
	MY_ASSERT(sscanf(words[0],"%f",&prob));
	if (howmany==(Order+2)){
		MY_ASSERT(sscanf(words[Order+1],"%f",&bow));
	}else{
		bow=0.0; //this is log10prob=0 for implicit backoff
	}
	return 1;
}

void exit_error(int err, const std::string &msg){
	if (msg != "") {
		VERBOSE(0,msg+"\n";);
	}
	else{
		switch(err){
			case IRSTLM_NO_ERROR:
				VERBOSE(0,"Everything OK\n");
				break;
			case IRSTLM_ERROR_GENERIC:
				VERBOSE(0,"Generic error\n");
				break;
			case IRSTLM_ERROR_IO:
				VERBOSE(0,"Input/Output error\n");
				break;
			case IRSTLM_ERROR_MEMORY:
				VERBOSE(0,"Allocation memory error\n");
				break;
			case IRSTLM_ERROR_DATA:
				VERBOSE(0,"Data format error\n");
				break;
			case IRSTLM_ERROR_MODEL:
				VERBOSE(0,"Model computation error\n");
				break;
			default:
				VERBOSE(0,"Undefined error\n");
				break;
		}
	}
	exit(err);
};

/*
#ifdef MY_ASSERT_FLAG
#if MY_ASSERT_FLAG>0
#undef MY_ASSERT(x)
#define MY_ASSERT(x) do { assert(x); } while (0)
#else
#define MY_ASSERT(x) { UNUSED(x); }
#endif
#else
#define MY_ASSERT(x) { UNUSED(x); }
#endif
*/

/** assert macros e functions**/
#ifdef MY_ASSERT_FLAG
#if MY_ASSERT_FLAG==0
#undef MY_ASSERT_FLAG
#endif
#endif

#ifdef MY_ASSERT_FLAG
void MY_ASSERT(bool x) { assert(x); }
#else
void MY_ASSERT(bool x) { UNUSED(x); }
#endif


/** trace macros and functions**/
/** verbose macros and functions**/

#ifdef TRACE_LEVEL
//int tracelevel=TRACE_LEVEL;
const int tracelevel=TRACE_LEVEL;
#else
//int tracelevel=0;
const int tracelevel=0;
#endif


namespace irstlm {
	void* reallocf(void *ptr, size_t size){
		void *p=realloc(ptr,size);
		
		if (p)
		{
			return p;
		}
		else
		{
			free(ptr);
			return NULL;
		}
	}
	
}

