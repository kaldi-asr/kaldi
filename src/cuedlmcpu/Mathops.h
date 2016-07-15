#ifndef _MATHOPS_H__
#define _MATHOPS_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "DataType.h"
using namespace std;

class CuedMatrix
{
	// number of row and column
	Ulint32 n_row, n_col;
	// data point
	int **data;
    int *dataptr;   // real date point
    void dimension (Ulint32 nrow, Ulint32 ncol);
    void Init ();
public:
	CuedMatrix ();
	CuedMatrix (Ulint32 nrow, Ulint32 ncol);
    CuedMatrix (CuedMatrix &mat, bool copy = false);
    void AllocMem ();
	~CuedMatrix ();
    void freeMem ();
	inline Ulint32 Getnrows () { return n_row; }
	inline Ulint32 Getncols () { return n_col; }
	int *operator [] (Ulint32 icol) { return data[icol]; }
	const int* operator [] (Ulint32 icol) const { return data[icol]; }
    int ** Getdatapoint () { return data;}
    void dump ();    // dump the whole matrix
};
#endif
