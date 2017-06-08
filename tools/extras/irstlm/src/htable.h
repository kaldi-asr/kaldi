// $Id: htable.h 3680 2010-10-13 09:10:21Z bertoldi $

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

#ifndef MF_HTABLE_H
#define MF_HTABLE_H

using namespace std;

#include <iostream>
#include <string>
#include <cstring>
#include "mempool.h"

#define Prime1                 37
#define Prime2                 1048583
#define BlockSize              100

typedef unsigned int address;

// Fast arithmetic, relying on powers of 2,
// and on pre-processor concatenation property
//use as
template <class T>
struct entry {
	T                 key;
	entry*           next;  // secret from user
};


typedef enum {HT_FIND,    //!< search: find an entry
	HT_ENTER,   //!< search: enter an entry
	HT_INIT,    //!< scan: start scan
	HT_CONT     //!< scan: continue scan
} HT_ACTION;

//!T is the type of the key and  should be (int*) or (char*)
template <class T>
class htable
{
	int        size;            //!< table size
	int      keylen;            //!< key length
	entry<T>   **table;            //!< hash table
	int      scan_i;            //!< scan support
	entry<T>   *scan_p;            //!< scan support
	// statistics
	long       keys;            //!< # of entries
	long   accesses;            //!< # of accesses
	long collisions;            //!< # of collisions
	
	mempool  *memory;           //!<  memory pool
	
public:
	
	//! Creates an hash table
	htable(int n,int kl=0);
	
	//! Destroys an and hash table
	~htable();
	
	void set_keylen(int kl);
	
	//! Computes the hash function
	address Hash(const T key);
	
	//! Compares the keys of two entries
	int Comp(const T Key1, const T Key2) const;
	
	//! Searches for an item
	T find(T item);
	T insert(T item);
	
	//! Scans the content
	T scan(HT_ACTION action);
	
	//! Prints statistics
	void stat() const ;
	
	//! Print a map of memory use
	void map(std::ostream& co=std::cout, int cols=80);
	
	//! Returns amount of used memory
	int used() const {
		return size * sizeof(entry<T> **) + memory->used();
	}
	
};



template <class T>
htable<T>::htable(int n,int kl)
{
	
	memory=new mempool( sizeof(entry<T>) , BlockSize );
	
	table = new entry<T>* [ size=n ];
	
	memset(table,0,sizeof(entry<T> *) * n );
	
	set_keylen(kl);
	
	keys = accesses = collisions = 0;
}

template <class T>
htable<T>::~htable()
{
	delete []table;
	delete memory;
}

template <class T>
T htable<T>::find(T key)
{
//	std::cerr << "T htable<T>::find(T key) size:" << size << std::endl;
	address    h;
	entry<T>  *q,**p;
	
	accesses++;
	
	h = Hash(key);
//	std::cerr << "T htable<T>::find(T key) h:" << h << std::endl;
	
	p=&table[h%size];
	q=*p;
	
	/* Follow collision chain */
	while (q != NULL && Comp(q->key,key)) {
		p = &(q->next);
		q = q->next;
		
		collisions++;
	}
	
	if (q != NULL) return q->key;    /* found */
	
	return NULL;
}

template <class T>
T htable<T>::insert(T key)
{
	address    h;
	entry<T>  *q,**p;
	
	accesses++;
	
	h = Hash(key);
	
	p=&table[h%size];
	q=*p;
	
	/* Follow collision chain */
	while (q != NULL && Comp(q->key,key)) {
		p = &(q->next);
		q = q->next;
		
		collisions++;
	}
	
	if (q != NULL) return q->key;    /* found */
	
	/* not found   */
	if ((q = (entry<T> *)memory->allocate()) == NULL) /* no room   */
		return NULL;
	
	/* link into chain      */
	*p = q;
	
	/* Initialize new element */
	q->key = key;
	q->next = NULL;
	keys++;
	
	return q->key;
}

template <class T>
T htable<T>::scan(HT_ACTION action)
{
	if (action == HT_INIT) {
		scan_i=0;
		scan_p=table[0];
		return NULL;
	}
	
	// if scan_p==NULL go to the first non null pointer
	while ((scan_p==NULL) && (++scan_i<size)) scan_p=table[scan_i];
	
	if (scan_p!=NULL) {
		T k = scan_p->key;
		scan_p=(entry<T> *)scan_p->next;
		return k;
	};
	
	return NULL;
}


template <class T>
void htable<T>::map(ostream& co,int cols)
{
	
	entry<T> *p;
	char* img=new char[cols+1];
	
	img[cols]='\0';
	memset(img,'.',cols);
	
	co << "htable memory map: . (0 items), - (<5), # (>5)\n";
	
	for (int i=0; i<size; i++) {
		int n=0;
		p=table[i];
		
		while(p!=NULL) {
			n++;
			p=(entry<T> *)p->next;
		};
		
		if (i && (i % cols)==0) {
			co << img << "\n";
			memset(img,'.',cols);
		}
		
		if (n>0)
			img[i % cols]=n<=5?'-':'#';
		
	}
	
	img[size % cols]='\0';
	co << img << "\n";
	
	delete []img;
}

template <class T>
void htable<T>::stat() const
{
	cerr << "htable class statistics\n";
	cerr << "size " << size
	<< " keys " << keys
	<< " acc " << accesses
	<< " coll " << collisions
	<< " used memory " << used()/1024 << "Kb\n";
};

#endif



