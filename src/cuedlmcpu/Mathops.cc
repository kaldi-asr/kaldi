#include "Mathops.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void CuedMatrix::Init ()
{
    n_row = 0;
    n_col = 0;
    data = NULL;
    dataptr = NULL;
}

void CuedMatrix::AllocMem ()
{
    if (n_row == 0 || n_col == 0)
    {
        cout << "ERROR: the number of row or column is 0 when trying to allocate memeory for matrx" << endl;
        exit (0);
    }
    if (data || dataptr)
    {
        cout << "Warning: the data and dataptr should be 0 before allocating memory" << endl;
    }
    Ulint32 size_data = n_col * sizeof (int *);
    data = (int **) malloc (size_data);
    memset (data, 0, size_data);
    Ulint32 size_dataptr = n_row * n_col * sizeof (int);
    dataptr = (int *) malloc (size_dataptr);
    memset (dataptr, 0, size_dataptr);
    for (Ulint32 i = 0; i < n_col; i ++)
    {
        data[i] = dataptr + i * n_row;
    }
}

// initialize the CuedMatrix
CuedMatrix::CuedMatrix ()
{
    Init ();
}
CuedMatrix::CuedMatrix (Ulint32 nrow, Ulint32 ncol)
{
    Init ();
    n_row = nrow;
    n_col = ncol;
    Ulint32 size_data = n_col * sizeof (int *);
    data = (int **) malloc (size_data);
    memset (data,0, size_data);
    Ulint32 size_dataptr = n_row * n_col * sizeof (int);
    dataptr = (int *)malloc (size_dataptr);
    memset (dataptr, 0, size_dataptr);
	for (Ulint32 i = 0; i < n_col; i ++)
	{
		data[i] = dataptr + i * n_row;
	}
}
CuedMatrix::CuedMatrix (CuedMatrix &mat, bool copy)
{
    Init ();
    dimension (mat.Getnrows() , mat.Getncols());
    AllocMem ();
    if (copy)
    {
        for (Ulint32 i = 0; i < n_row; i ++)
        {
            for (Ulint32 j = 0; j < n_col; j ++)
            {
                data[i][j] = mat[i][j];
            }
        }
    }
    else
    {
        data = mat.Getdatapoint ();
    }
}

CuedMatrix::~CuedMatrix ()
{
	if (data != NULL)
	{
        free (data);
        data = NULL;
	}
    if (dataptr != NULL)
    {
        free (dataptr);
        dataptr = NULL;
    }
}

void CuedMatrix::freeMem ()   // free memory.
{
	if (data != NULL)
	{
        free (data);
        data = NULL;
	}
    if (dataptr != NULL)
    {
        free (dataptr);
        dataptr = NULL;
    }
}

void CuedMatrix::dimension (Ulint32 nrow, Ulint32 ncol)
{
    n_row = nrow;
    n_col = ncol;
}

void CuedMatrix::dump ()
{
    for (int i = 0; i < n_col; i ++)
    {
        printf ("col %d\t", i);
        for (int j = 0; j < n_row; j++ )
        {
            printf ("%d ", data[i][j]);
        }
        printf ("\n");
    }
}
