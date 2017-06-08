// $Id: mempool.cpp 302 2009-08-25 13:04:13Z nicolabertoldi $

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

// An efficient memory pool manager
// by M. Federico
// Copyright Marcello Federico, ITC-irst, 1998

#include <stdio.h>
#include <cstring>
#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <ostream>
#include "util.h"
#include "mempool.h"

using namespace std;

/*! The pool contains:
 - entries of size is
 - tables for bs entries
 */


mempool::mempool(int is, int bs)
{
	
	// item size must be multiple of memory alignment step (4 bytes)
	// example: is is=9  becomes i=12 (9 + 4 - 9 %4 )
	
	is=(is>(int)sizeof(char *)?is:0);
	
	is=is + sizeof(char *) - (is % sizeof(char *));
	
	item_size  = is;
	
	block_size = bs;
	
	true_size  = is * bs;
	
	block_list = new memnode;
	
	block_list->block = new char[true_size];
	
	memset(block_list->block,'0',true_size);
	
	block_list->next  = 0;
	
	blocknum = 1;
	
	entries  = 0;
	
	// build free list
	
	char *ptr = free_list = block_list->block;
	
	for (int i=0; i<block_size-1; i++) {
		*(char **)ptr= ptr + item_size;
		ptr+=item_size;
	}
	*(char **)ptr = NULL; //last item
	
}


char * mempool::allocate()
{
	
	char *ptr;
	
	if (free_list==NULL) {
		memnode *new_block = new memnode;
		
		new_block->block = new char[true_size];
		
		//memset(new_block->block,'0',true_size);
		
		new_block->next  = block_list;
		
		block_list=new_block; // update block list
		
		/* update  free list */
		
		ptr = free_list = block_list->block;
		
		for (int i=0; i<block_size-1; i++) {
			*(char **)ptr = ptr + item_size;
			ptr = ptr + item_size;
		}
		
		*(char **)ptr=NULL;
		
		blocknum++;
	}
	
	MY_ASSERT(free_list);
	
	ptr = free_list;
	
	free_list=*(char **)ptr;
	
	*(char **)ptr=NULL; // reset the released item
	
	entries++;
	
	return ptr;
	
}


int mempool::free(char* addr)
{
	
	// do not check if it belongs to this pool !!
	/*
	 memnode  *list=block_list;
	 while ((list != NULL) &&
	 ((addr < list->block) ||
	 (addr >= (list->block + true_size))))
	 list=list->next;
	 
	 if ((list==NULL) || (((addr - list->block) % item_size)!=0))
	 {
	 //cerr << "mempool::free-> addr does not belong to this pool\n";
	 return 0;
	 }
	 */
	
	*(char **)addr=free_list;
	free_list=addr;
	
	entries--;
	
	return 1;
}


mempool::~mempool()
{
	memnode *ptr;
	
	while (block_list !=NULL) {
		ptr=block_list->next;
		delete [] block_list->block;
		delete block_list;
		block_list=ptr;
	}
	
}

void mempool::map (ostream& co)
{
	
	co << "mempool memory map:\n";
	//percorri piu` volte la lista libera
	
	memnode *bl=block_list;
	char *fl=free_list;
	
	char* img=new char[block_size+1];
	img[block_size]='\0';
	
	while (bl !=NULL) {
		
		memset(img,'#',block_size);
		
		fl=free_list;
		while (fl != NULL) {
			if ((fl >= bl->block)
					&&
					(fl < bl->block + true_size)) {
				img[(fl-bl->block)/item_size]='-';
			}
			
			fl=*(char **)fl;
		}
		
		co << img << "\n";
		bl=bl->next;
	}
	delete [] img;
}

void mempool::stat()
{
	
	VERBOSE(1, "mempool class statistics\n"
					<< "entries " << entries
					<< " blocks " << blocknum
					<< " used memory " << (blocknum * true_size)/1024 << " Kb\n");
}



strstack::strstack(int bs)
{
	
	size=bs;
	list=new memnode;
	
	list->block=new char[size];
	
	list->next=0;
	
	memset(list->block,'\0',size);
	idx=0;
	
	waste=0;
	memory=size;
	entries=0;
	blocknum=1;
	
}


void strstack::stat()
{
	
	VERBOSE(1, "strstack class statistics\n"
					<< "entries " << entries
					<< " blocks " << blocknum
					<< " used memory " << memory/1024 << " Kb\n");
}


const char *strstack::push(const char *s)
{
	int len=strlen(s);
	
	if ((len+1) >= size) {
		exit_error(IRSTLM_ERROR_DATA, "strstack::push string is too long");
	};
	
	if ((idx+len+1) >= size) {
		//append a new block
		//there must be space to
		//put the index after
		//the word
		
		waste+=size-idx;
		blocknum++;
		memory+=size;
		
		memnode* nd=new memnode;
		nd->block=new char[size];
		nd->next=list;
		
		list=nd;
		
		memset(list->block,'\0',size);
		
		idx=0;
		
	}
	
	// append in current block
	
	strcpy(&list->block[idx],s);
	
	idx+=len+1;
	
	entries++;
	
	return &list->block[idx-len-1];
	
}


const char *strstack::pop()
{
	
	if (list==0) return 0;
	
	if (idx==0) {
		
		// free this block and go to next
		
		memnode *ptr=list->next;
		
		delete [] list->block;
		delete list;
		
		list=ptr;
		
		if (list==0)
			return 0;
		else
			idx=size-1;
	}
	
	//go back to first non \0
	while (idx>0)
		if (list->block[idx--]!='\0')
			break;
	
	//go back to first \0
	while (idx>0)
		if (list->block[idx--]=='\0')
			break;
	
	entries--;
	
	if (list->block[idx+1]=='\0') {
		idx+=2;
		memset(&list->block[idx],'\0',size-idx);
		return &list->block[idx];
	} else {
		idx=0;
		memset(&list->block[idx],'\0',size);
		return &list->block[0];
	}
}


const char *strstack::top()
{
	
	int tidx=idx;
	memnode *tlist=list;
	
	if (tlist==0) return 0;
	
	if (idx==0) {
		
		tlist=tlist->next;
		
		if (tlist==0) return 0;
		
		tidx=size-1;
	}
	
	//go back to first non \0
	while (tidx>0)
		if (tlist->block[tidx--]!='\0')
			break;
	
	//aaa\0bbb\0\0\0\0
	
	//go back to first \0
	while (tidx>0)
		if (tlist->block[tidx--]=='\0')
			break;
	
	if (tlist->block[tidx+1]=='\0') {
		tidx+=2;
		return &tlist->block[tidx];
	} else {
		tidx=0;
		return &tlist->block[0];
	}
	
}


strstack::~strstack()
{
	memnode *ptr;
	while (list !=NULL) {
		ptr=list->next;
		delete [] list->block;
		delete list;
		list=ptr;
	}
}


storage::storage(int maxsize,int blocksize)
{
	newmemory=0;
	newcalls=0;
	setsize=maxsize;
	poolsize=blocksize; //in bytes
	poolset=new mempool* [setsize+1];
	for (int i=0; i<=setsize; i++)
		poolset[i]=NULL;
}


storage::~storage()
{
	for (int i=0; i<=setsize; i++)
		if (poolset[i])
			delete poolset[i];
	delete [] poolset;
}

char *storage::allocate(int size)
{
	
	if (size<=setsize) {
		if (!poolset[size]) {
			poolset[size]=new mempool(size,poolsize/size);
		}
		return poolset[size]->allocate();
	} else {
		
		newmemory+=size+8;
		newcalls++;
		char* p=(char *)calloc(sizeof(char),size);
		if (p==NULL) {
			exit_error(IRSTLM_ERROR_MEMORY, "storage::alloc insufficient memory");
		}
		return p;
	}
}

char *storage::reallocate(char *oldptr,int oldsize,int newsize)
{
	
	char *newptr;
	
	MY_ASSERT(newsize>oldsize);
	
	if (oldsize<=setsize) {
		if (newsize<=setsize) {
			if (!poolset[newsize])
				poolset[newsize]=new mempool(newsize,poolsize/newsize);
			newptr=poolset[newsize]->allocate();
			memset((char*)newptr,0,newsize);
		} else
			newptr=(char *)calloc(sizeof(char),newsize);
		
		if (oldptr && oldsize) {
			memcpy(newptr,oldptr,oldsize);
			poolset[oldsize]->free(oldptr);
		}
	} else {
		newptr=(char *)realloc(oldptr,newsize);
		if (newptr==oldptr)
			cerr << "r\b";
		else
			cerr << "a\b";
	}
	if (newptr==NULL) {
		exit_error(IRSTLM_ERROR_MEMORY,"storage::realloc insufficient memory");
	}
	
	return newptr;
}

int storage::free(char *addr,int size)
{
	
	/*
	 while(size<=setsize){
	 if (poolset[size] && poolset[size]->free(addr))
	 break;
	 size++;
	 }
	 */
	
	if (size>setsize)
		return free(addr),1;
	else {
		poolset[size] && poolset[size]->free(addr);
	}
	return 1;
}


void storage::stat()
{
	IFVERBOSE(1){
		int used=0;
		int memory=sizeof(char *) * setsize;
		int waste=0;
		
		for (int i=0; i<=setsize; i++)
			if (poolset[i]) {
				used++;
				memory+=poolset[i]->used();
				waste+=poolset[i]->wasted();
			}
		
		VERBOSE(1, "storage class statistics\n"
						<< "alloc entries " << newcalls
						<< " used memory " << newmemory/1024 << "Kb\n"
						<< "mpools " << setsize
						<< " active  " << used
						<< " used memory " << memory/1024 << "Kb"
						<< " wasted " << waste/1024 << "Kb\n");
	}
}


